"""
Evaluation runner for AgentXAI.

Computes five metrics:

1. Task performance — accuracy, per-option confusion matrix (A–E), and a
   10-bin reliability diagram for confidence calibration.

2. Sufficiency (XAI) — re-run the Synthesizer with only the top-1
   responsible agent's memory; zero out all other specialists. Sufficiency
   score = fraction of tasks where the final diagnosis is unchanged.

3. Necessity (XAI) — re-run the Synthesizer with the top-1 agent's memory
   zeroed out and all others kept. Necessity score = fraction of tasks where
   the final diagnosis changes.

4. Stability — for N sampled tasks, generate a minor LLM rephrasing of the
   patient case, re-run the full pipeline, and compute the Spearman rank
   correlation of per-agent responsibility scores between the two runs.
   Reports mean ± std across sampled tasks.

5. Faithfulness — for each task, zero out the memory of the agent that owns
   the root_cause_event (neutral baseline intervention). Faithfulness score =
   fraction of tasks where the outcome actually changes.

Output
------
eval/results_<timestamp>.json  — machine-readable report (all metrics)
eval/results_<timestamp>.md    — Markdown summary (one section per metric)

CLI
---
    python -m eval.evaluate --limit 1500 --samples-for-stability 100
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agentxai.data.load_medqa import load_medqa_us_all, make_splits
from agentxai.data.schemas import AgentXAIRecord, AccountabilityReport
from run_pipeline import Pipeline

_log = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Spearman rank correlation (stdlib-only, no scipy dependency)
# ---------------------------------------------------------------------------

def _spearman(x: List[float], y: List[float]) -> float:
    """Return Spearman rank correlation; nan when undefined (< 2 values or zero variance)."""
    n = len(x)
    if n < 2:
        return float("nan")

    def _ranks(arr: List[float]) -> List[float]:
        indexed = sorted(enumerate(arr), key=lambda t: t[1])
        result = [0.0] * n
        i = 0
        while i < n:
            j = i + 1
            while j < n and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j + 1) / 2.0  # 1-based average rank for ties
            for k in range(i, j):
                result[indexed[k][0]] = avg_rank
            i = j
        return result

    rx, ry = _ranks(x), _ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if dx == 0.0 or dy == 0.0:
        return float("nan")
    return num / (dx * dy)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

_REPHRASE_PROMPT = (
    "Rephrase the following clinical vignette without changing any clinical "
    "meaning, facts, or details. Use different wording and sentence structure "
    "while preserving all medical information exactly.\n\n"
    "Vignette:\n{case}\n\n"
    "Return ONLY the rephrased vignette text, with no preamble or commentary."
)


def _rephrase_case(llm: Any, case: str) -> str:
    """Use the pipeline LLM to rephrase a patient case preserving clinical meaning."""
    if llm is None or not case.strip():
        return ""
    prompt = _REPHRASE_PROMPT.format(case=case)
    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(str(block))
            return "".join(parts).strip()
        return str(content).strip()
    except Exception as exc:
        _log.warning("Rephrase LLM call failed: %s", exc)
        return ""


def _extract_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(
            str(b.get("text", "")) if isinstance(b, dict) else str(b)
            for b in content
        )
    return str(content)


# ---------------------------------------------------------------------------
# Accountability report helpers
# ---------------------------------------------------------------------------

def _top_agent(report: AccountabilityReport) -> Optional[str]:
    scores = report.agent_responsibility_scores
    if not scores:
        return None
    return max(scores, key=lambda a: scores[a])


def _specialist_agents(report: AccountabilityReport) -> List[str]:
    return list(report.agent_responsibility_scores.keys())


# ---------------------------------------------------------------------------
# Metric 1: Task performance
# ---------------------------------------------------------------------------

def compute_task_performance(results: List[AgentXAIRecord]) -> Dict[str, Any]:
    """Accuracy, per-option confusion matrix (A–E), and calibration bins."""
    if not results:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_total": 0,
            "confusion_matrix": {"labels": [], "matrix": []},
            "calibration_bins": [],
        }

    n_correct = sum(1 for r in results if r.system_output.get("correct", False))
    accuracy = n_correct / len(results)

    # Confusion matrix over the five canonical option letters.
    letters = ["A", "B", "C", "D", "E"]
    letter_idx = {ltr: i for i, ltr in enumerate(letters)}
    matrix = [[0] * len(letters) for _ in range(len(letters))]
    for r in results:
        gt = r.ground_truth.get("correct_answer", "")
        pred = r.system_output.get("predicted_letter", "")
        if gt in letter_idx and pred in letter_idx:
            matrix[letter_idx[gt]][letter_idx[pred]] += 1

    # Reliability diagram: 10 equal-width bins over [0, 1] confidence.
    bins: List[List[bool]] = [[] for _ in range(10)]
    for r in results:
        conf = float(r.system_output.get("confidence", 0.0) or 0.0)
        conf = max(0.0, min(1.0, conf))
        correct = bool(r.system_output.get("correct", False))
        bins[min(int(conf * 10), 9)].append(correct)

    calibration_bins = []
    for i, b in enumerate(bins):
        calibration_bins.append({
            "bin_lower": round(i / 10, 1),
            "bin_upper": round((i + 1) / 10, 1),
            "mean_confidence": round((i + 0.5) / 10, 2),
            "fraction_correct": round(sum(b) / len(b), 4) if b else None,
            "count": len(b),
        })

    return {
        "accuracy": round(accuracy, 4),
        "n_correct": n_correct,
        "n_total": len(results),
        "confusion_matrix": {"labels": letters, "matrix": matrix},
        "calibration_bins": calibration_bins,
    }


# ---------------------------------------------------------------------------
# Metric 2: Sufficiency (XAI)
# ---------------------------------------------------------------------------

def compute_sufficiency(
    pipeline: Pipeline,
    results: List[AgentXAIRecord],
) -> Dict[str, Any]:
    """
    Re-run Synthesizer with ONLY the top-1 responsible agent's memory (zero
    out all other specialists). Sufficiency = fraction of tasks where the
    final diagnosis is unchanged from the original run.
    """
    unchanged = 0
    total = 0
    errors = 0

    for r in results:
        report = r.xai_data.accountability_report
        if report is None:
            continue
        top = _top_agent(report)
        if top is None:
            continue
        snapshot = pipeline._snapshots.get(r.task_id)
        if snapshot is None:
            continue

        others = [a for a in _specialist_agents(report) if a != top]
        if not others:
            # Only one specialist — trivially sufficient.
            unchanged += 1
            total += 1
            continue

        original_dx = r.system_output.get("final_diagnosis", "")
        overrides: Dict[str, Any] = {"agent_memory": {a: {} for a in others}}

        try:
            cf = pipeline.resume_from(snapshot, overrides)
            if cf.get("final_diagnosis", "") == original_dx:
                unchanged += 1
        except Exception as exc:
            _log.warning("Sufficiency re-run failed for %s: %s", r.task_id, exc)
            errors += 1

        total += 1

    return {
        "sufficiency_score": round(unchanged / total, 4) if total > 0 else 0.0,
        "n_unchanged": unchanged,
        "n_total": total,
        "n_errors": errors,
    }


# ---------------------------------------------------------------------------
# Metric 3: Necessity (XAI)
# ---------------------------------------------------------------------------

def compute_necessity(
    pipeline: Pipeline,
    results: List[AgentXAIRecord],
) -> Dict[str, Any]:
    """
    Re-run Synthesizer with the top-1 agent's memory zeroed out (keep all
    others). Necessity = fraction of tasks where the final diagnosis changes.
    """
    changed = 0
    total = 0
    errors = 0

    for r in results:
        report = r.xai_data.accountability_report
        if report is None:
            continue
        top = _top_agent(report)
        if top is None:
            continue
        snapshot = pipeline._snapshots.get(r.task_id)
        if snapshot is None:
            continue

        original_dx = r.system_output.get("final_diagnosis", "")
        overrides: Dict[str, Any] = {"agent_memory": {top: {}}}

        try:
            cf = pipeline.resume_from(snapshot, overrides)
            if cf.get("final_diagnosis", "") != original_dx:
                changed += 1
        except Exception as exc:
            _log.warning("Necessity re-run failed for %s: %s", r.task_id, exc)
            errors += 1

        total += 1

    return {
        "necessity_score": round(changed / total, 4) if total > 0 else 0.0,
        "n_changed": changed,
        "n_total": total,
        "n_errors": errors,
    }


# ---------------------------------------------------------------------------
# Metric 4: Stability
# ---------------------------------------------------------------------------

def compute_stability(
    pipeline: Pipeline,
    paired: List[Tuple[Dict, AgentXAIRecord]],
    n_samples: int = 100,
) -> Dict[str, Any]:
    """
    Sample n_samples tasks, rephrase each patient case via LLM, re-run the
    full pipeline on the rephrased case, and compute the Spearman rank
    correlation of per-agent responsibility scores between the two runs.
    Reports mean ± std across successfully computed correlations.
    """
    n = min(n_samples, len(paired))
    sampled = random.sample(paired, n)

    correlations: List[float] = []
    skipped = 0
    errors = 0

    for i, (rec, orig_result) in enumerate(sampled):
        orig_case = orig_result.input.get("patient_case", "")
        rephrased = _rephrase_case(pipeline.llm, orig_case)
        if not rephrased:
            skipped += 1
            continue

        reph_rec = dict(rec)
        reph_rec["question"] = rephrased

        try:
            reph_result = pipeline.run_task(reph_rec)
        except Exception as exc:
            _log.warning("Stability pipeline re-run failed (sample %d): %s", i, exc)
            errors += 1
            continue

        orig_report = orig_result.xai_data.accountability_report
        reph_report = reph_result.xai_data.accountability_report
        if orig_report is None or reph_report is None:
            skipped += 1
            continue

        orig_scores = orig_report.agent_responsibility_scores
        reph_scores = reph_report.agent_responsibility_scores
        agents = sorted(set(orig_scores) | set(reph_scores))
        if len(agents) < 2:
            skipped += 1
            continue

        orig_vec = [float(orig_scores.get(a, 0.0)) for a in agents]
        reph_vec = [float(reph_scores.get(a, 0.0)) for a in agents]
        corr = _spearman(orig_vec, reph_vec)
        if not math.isnan(corr):
            correlations.append(corr)
        else:
            skipped += 1

        if (i + 1) % 10 == 0:
            _log.info("  stability: %d / %d done", i + 1, n)

    mean_corr: Optional[float]
    std_corr: Optional[float]
    if correlations:
        mean_corr = sum(correlations) / len(correlations)
        var = sum((c - mean_corr) ** 2 for c in correlations) / len(correlations)
        std_corr = math.sqrt(var)
    else:
        mean_corr = std_corr = None

    return {
        "stability_mean_spearman": round(mean_corr, 4) if mean_corr is not None else None,
        "stability_std_spearman": round(std_corr, 4) if std_corr is not None else None,
        "n_sampled": n,
        "n_computed": len(correlations),
        "n_skipped": skipped,
        "n_errors": errors,
    }


# ---------------------------------------------------------------------------
# Metric 5: Faithfulness
# ---------------------------------------------------------------------------

def compute_faithfulness(
    pipeline: Pipeline,
    results: List[AgentXAIRecord],
) -> Dict[str, Any]:
    """
    For each task, zero out the memory of the agent that owns the
    root_cause_event (neutral baseline intervention) and re-run the
    Synthesizer. Faithfulness = fraction of tasks where the outcome changes.
    """
    changed = 0
    total = 0
    errors = 0

    for r in results:
        report = r.xai_data.accountability_report
        if report is None or not report.root_cause_event_id:
            continue

        # Find which agent owns the root cause trajectory event.
        root_agent: Optional[str] = None
        for event in r.xai_data.trajectory:
            if event.event_id == report.root_cause_event_id:
                root_agent = event.agent_id
                break

        # Only specialist agents can be intervened on via memory zeroing.
        if root_agent not in ("specialist_a", "specialist_b"):
            continue

        snapshot = pipeline._snapshots.get(r.task_id)
        if snapshot is None:
            continue

        original_dx = r.system_output.get("final_diagnosis", "")
        overrides: Dict[str, Any] = {"agent_memory": {root_agent: {}}}

        try:
            cf = pipeline.resume_from(snapshot, overrides)
            if cf.get("final_diagnosis", "") != original_dx:
                changed += 1
        except Exception as exc:
            _log.warning("Faithfulness re-run failed for %s: %s", r.task_id, exc)
            errors += 1

        total += 1

    return {
        "faithfulness_score": round(changed / total, 4) if total > 0 else 0.0,
        "n_changed": changed,
        "n_total": total,
        "n_errors": errors,
    }


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _safe(v: Any) -> Any:
    """Recursively make a value JSON-safe (convert nan/inf → None)."""
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, dict):
        return {k: _safe(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_safe(x) for x in v]
    return v


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _md_report(report: Dict[str, Any]) -> str:
    cfg = report["config"]
    tp = report["task_performance"]
    suf = report["sufficiency"]
    nec = report["necessity"]
    stab = report["stability"]
    faith = report["faithfulness"]

    lines: List[str] = [
        "# AgentXAI Evaluation Report",
        "",
        f"**Generated:** {report['timestamp']}",
        f"**Records evaluated:** {cfg['limit']}  "
        f"**Stability samples:** {cfg['samples_for_stability']}",
        "",
    ]

    # --- Task performance ---
    lines += [
        "## 1. Task Performance",
        "",
        f"**Accuracy:** {tp['accuracy']:.1%}  ({tp['n_correct']} / {tp['n_total']} correct)",
        "",
        "### Confidence Calibration (reliability diagram, 10 bins)",
        "",
        "| Bin | Mean conf | Fraction correct | Count |",
        "|-----|-----------|-----------------|-------|",
    ]
    for b in tp["calibration_bins"]:
        fc = f"{b['fraction_correct']:.3f}" if b["fraction_correct"] is not None else "—"
        lines.append(
            f"| {b['bin_lower']:.1f}–{b['bin_upper']:.1f}"
            f" | {b['mean_confidence']:.2f}"
            f" | {fc}"
            f" | {b['count']} |"
        )

    labels = tp["confusion_matrix"]["labels"]
    matrix = tp["confusion_matrix"]["matrix"]
    lines += [
        "",
        "### Per-Option Confusion Matrix (rows = ground truth, cols = predicted)",
        "",
        "| truth \\ pred | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for i, row in enumerate(matrix):
        lines.append(f"| **{labels[i]}** | " + " | ".join(str(c) for c in row) + " |")

    # --- Sufficiency ---
    lines += [
        "",
        "## 2. Sufficiency (XAI)",
        "",
        f"**Sufficiency score:** {suf['sufficiency_score']:.1%}",
        f"- Diagnosis **unchanged** in {suf['n_unchanged']} / {suf['n_total']} tasks "
        f"when re-run with only the top-1 responsible agent's memory.",
        f"- Errors: {suf['n_errors']}",
        "",
    ]

    # --- Necessity ---
    lines += [
        "## 3. Necessity (XAI)",
        "",
        f"**Necessity score:** {nec['necessity_score']:.1%}",
        f"- Diagnosis **changed** in {nec['n_changed']} / {nec['n_total']} tasks "
        f"after zeroing out the top-1 agent's memory.",
        f"- Errors: {nec['n_errors']}",
        "",
    ]

    # --- Stability ---
    mean_s = stab["stability_mean_spearman"]
    std_s = stab["stability_std_spearman"]
    mean_str = f"{mean_s:.4f}" if mean_s is not None else "N/A"
    std_str = f"{std_s:.4f}" if std_s is not None else "N/A"
    lines += [
        "## 4. Stability",
        "",
        f"**Mean Spearman ρ:** {mean_str} ± {std_str}",
        f"- Computed on {stab['n_computed']} / {stab['n_sampled']} sampled tasks "
        f"({stab['n_skipped']} skipped, {stab['n_errors']} errors).",
        "",
    ]

    # --- Faithfulness ---
    lines += [
        "## 5. Faithfulness",
        "",
        f"**Faithfulness score:** {faith['faithfulness_score']:.1%}",
        f"- Outcome **changed** in {faith['n_changed']} / {faith['n_total']} tasks "
        f"after zeroing out the root-cause agent's memory.",
        f"- Errors: {faith['n_errors']}",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    limit: int = 1500,
    samples_for_stability: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run the full evaluation suite and return the results dict.

    Parameters
    ----------
    limit:
        Maximum number of records from the eval split to process.
    samples_for_stability:
        Number of tasks randomly sampled for the stability metric.
    seed:
        Random seed used for make_splits() and stability sampling.
    """
    random.seed(seed)

    _log.info("Loading MedQA US (train + dev + test) …")
    all_records = load_medqa_us_all()
    _, eval_split, _ = make_splits(all_records, eval_size=1500, review_size=500, seed=seed)
    eval_records = eval_split[:limit]
    _log.info("Eval split size: %d  (limit=%d)", len(eval_split), limit)

    pipeline = Pipeline()

    # ------------------------------------------------------------------
    # Step 1 — Run the pipeline on every eval record.
    # Keep (record, result) pairs so stability can align them precisely.
    # ------------------------------------------------------------------
    _log.info("Running pipeline on %d eval records …", len(eval_records))
    paired: List[Tuple[Dict, AgentXAIRecord]] = []
    for i, rec in enumerate(eval_records):
        try:
            result = pipeline.run_task(rec)
            paired.append((rec, result))
        except Exception as exc:
            _log.error(
                "Pipeline failed on record %d (task_id=%s): %s",
                i,
                rec.get("task_id", "?"),
                exc,
            )
        if (i + 1) % 50 == 0 or (i + 1) == len(eval_records):
            _log.info("  %d / %d done", i + 1, len(eval_records))

    results = [p[1] for p in paired]
    _log.info("Pipeline done: %d / %d succeeded.", len(results), len(eval_records))

    # ------------------------------------------------------------------
    # Step 2 — Task performance
    # ------------------------------------------------------------------
    _log.info("Computing task performance …")
    task_perf = compute_task_performance(results)
    _log.info("  accuracy=%.1f%%", task_perf["accuracy"] * 100)

    # ------------------------------------------------------------------
    # Step 3 — Sufficiency
    # ------------------------------------------------------------------
    _log.info("Computing sufficiency …")
    suf = compute_sufficiency(pipeline, results)
    _log.info("  sufficiency_score=%.4f", suf["sufficiency_score"])

    # ------------------------------------------------------------------
    # Step 4 — Necessity
    # ------------------------------------------------------------------
    _log.info("Computing necessity …")
    nec = compute_necessity(pipeline, results)
    _log.info("  necessity_score=%.4f", nec["necessity_score"])

    # ------------------------------------------------------------------
    # Step 5 — Stability
    # ------------------------------------------------------------------
    _log.info("Computing stability (n_samples=%d) …", samples_for_stability)
    stab = compute_stability(pipeline, paired, n_samples=samples_for_stability)
    _log.info(
        "  stability_mean=%.4f  std=%.4f  computed=%d/%d",
        stab["stability_mean_spearman"] or 0.0,
        stab["stability_std_spearman"] or 0.0,
        stab["n_computed"],
        stab["n_sampled"],
    )

    # ------------------------------------------------------------------
    # Step 6 — Faithfulness
    # ------------------------------------------------------------------
    _log.info("Computing faithfulness …")
    faith = compute_faithfulness(pipeline, results)
    _log.info("  faithfulness_score=%.4f", faith["faithfulness_score"])

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {
        "timestamp": ts,
        "config": {
            "limit": limit,
            "samples_for_stability": samples_for_stability,
            "seed": seed,
        },
        "task_performance": task_perf,
        "sufficiency": suf,
        "necessity": nec,
        "stability": stab,
        "faithfulness": faith,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run AgentXAI evaluation and write JSON + Markdown reports."
    )
    p.add_argument(
        "--limit", type=int, default=1500,
        help="Max eval records to run (default: 1500).",
    )
    p.add_argument(
        "--samples-for-stability", type=int, default=100,
        help="Tasks sampled for stability metric (default: 100).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for split + stability sampling (default: 42).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=_EVAL_DIR,
        help="Directory for output reports (default: eval/).",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    report = run_evaluation(
        limit=args.limit,
        samples_for_stability=args.samples_for_stability,
        seed=args.seed,
    )

    ts = report["timestamp"]
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"results_{ts}.json"
    md_path = out_dir / f"results_{ts}.md"

    json_path.write_text(json.dumps(_safe(report), indent=2), encoding="utf-8")
    md_path.write_text(_md_report(report), encoding="utf-8")

    # Print summary to stdout.
    tp = report["task_performance"]
    suf = report["sufficiency"]
    nec = report["necessity"]
    stab = report["stability"]
    faith = report["faithfulness"]
    mean_s = stab["stability_mean_spearman"]

    print(f"\n=== AgentXAI Evaluation Results ({ts}) ===")
    print(f"  Task accuracy    : {tp['accuracy']:.1%}  ({tp['n_correct']}/{tp['n_total']})")
    print(f"  Sufficiency      : {suf['sufficiency_score']:.1%}")
    print(f"  Necessity        : {nec['necessity_score']:.1%}")
    if mean_s is not None:
        print(f"  Stability (ρ)   : {mean_s:.4f} ± {stab['stability_std_spearman']:.4f}")
    else:
        print("  Stability (ρ)   : N/A")
    print(f"  Faithfulness     : {faith['faithfulness_score']:.1%}")
    print(f"\n  JSON  → {json_path}")
    print(f"  MD    → {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
