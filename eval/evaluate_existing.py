"""
Recompute the 5 evaluation metrics on tasks already stored in the SQLite DB,
without re-running the forward pipeline. Saves ~2,800 LLM calls vs. the
canonical eval.evaluate runner.

Steps:
1. Load the most recent N tasks from the trajectory store.
2. Match each task's patient case against the MedQA splits to recover the
   true ground-truth letter, fix the buggy `correct_answer` / `correct`
   fields in-place, and write the fixes back to the DB (so the dashboard
   ✓/✗ is also corrected).
3. Reconstruct each task's pipeline snapshot from the stored XAI artefacts
   so the counterfactual metric functions can run.
4. Reuse the metric implementations in eval.evaluate to compute task
   performance + sufficiency + necessity + stability + faithfulness.
5. Write results_existing_<timestamp>.{json,md} into eval/.

Usage:
    python -m eval.evaluate_existing --limit 100 --samples-for-stability 10
"""
from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sqlalchemy import text

from agentxai.data.load_medqa import load_medqa_us
from agentxai.data.schemas import AgentXAIRecord
from agentxai.store.trajectory_store import TrajectoryStore

# Reuse the metric implementations and reporting from the canonical eval.
from eval.evaluate import (
    compute_task_performance,
    compute_sufficiency,
    compute_necessity,
    compute_stability,
    compute_faithfulness,
    _md_report,
    _safe,
)
from run_pipeline import Pipeline, _resolve_correct_letter


_log = logging.getLogger(__name__)
_EVAL_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Question-text → MedQA record index (so we can recover the true answer letter)
# ---------------------------------------------------------------------------

def _build_medqa_index() -> Dict[str, Dict]:
    idx: Dict[str, Dict] = {}
    for split in ("train", "dev", "test"):
        try:
            for r in load_medqa_us(split):
                idx[r["question"].strip()] = r
        except FileNotFoundError:
            continue
    return idx


# ---------------------------------------------------------------------------
# Per-task: fix scoring + rebuild pipeline snapshot
# ---------------------------------------------------------------------------

def _fix_record_scoring(
    rec: AgentXAIRecord,
    medqa_index: Dict[str, Dict],
    store: TrajectoryStore,
) -> Dict[str, Any]:
    """
    If the task's patient case is in MedQA, recover the true letter and
    update record.ground_truth + record.system_output.correct in-place,
    then persist back to the store. Returns the original MedQA record (for
    stability) or a synthetic minimal one if no match was found.
    """
    case = (rec.input.get("patient_case") or "").strip()
    options = rec.input.get("options") or {}
    medqa_rec = medqa_index.get(case)

    if medqa_rec is not None:
        correct_letter = _resolve_correct_letter(medqa_rec)
        rec.ground_truth["correct_answer"] = correct_letter
        rec.ground_truth["answer_text"] = options.get(correct_letter, "") or medqa_rec.get("answer", "")

        pred_letter = (rec.system_output.get("predicted_letter") or "").strip().upper()
        rec.system_output["correct"] = bool(pred_letter and pred_letter == correct_letter)
        store.save_task(rec)
        return medqa_rec

    # Fallback: synth minimal record (no MedQA hit). Stability rephrasing
    # may still work; correct flag stays as whatever was previously stored.
    return {
        "question":   case,
        "options":    options,
        "answer":     rec.ground_truth.get("answer_text", ""),
        "answer_idx": rec.ground_truth.get("correct_answer", ""),
    }


def _build_snapshot_from_record(
    rec: AgentXAIRecord,
) -> Dict[str, Any]:
    """
    Rebuild the per-task snapshot dict that the counterfactual engine needs.
    Mirrors run_pipeline.Pipeline._build_snapshot but works directly off an
    already-loaded AgentXAIRecord (no fresh DB query required).
    """
    case = rec.input.get("patient_case") or ""
    options = dict(rec.input.get("options") or {})

    # Final per-agent memory state, reconstructed from the ordered diff log.
    mem_a: Dict[str, Any] = {}
    mem_b: Dict[str, Any] = {}
    for d in rec.xai_data.memory_diffs:
        if d.operation != "write":
            continue
        if d.agent_id == "specialist_a":
            mem_a[d.key] = d.value_after
        elif d.agent_id == "specialist_b":
            mem_b[d.key] = d.value_after

    tool_owner = {tc.tool_call_id: tc.called_by for tc in rec.xai_data.tool_calls}
    tool_name  = {tc.tool_call_id: tc.tool_name for tc in rec.xai_data.tool_calls}

    message_senders  = {m.message_id: m.sender        for m in rec.xai_data.messages}
    message_contents = {m.message_id: dict(m.content) for m in rec.xai_data.messages}

    return {
        "input_payload":    {"patient_case": case, "options": options},
        "agent_memory":     {"specialist_a": mem_a, "specialist_b": mem_b},
        "tool_owner":       tool_owner,
        "tool_name":        tool_name,
        "message_senders":  message_senders,
        "message_contents": message_contents,
        "original_output":  dict(rec.system_output or {}),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(limit: int, samples_for_stability: int, seed: int = 42) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")

    store = TrajectoryStore()
    pipeline = Pipeline()

    with store._engine.connect() as conn:
        rows = conn.execute(text(
            f"SELECT task_id FROM tasks ORDER BY created_at DESC LIMIT {limit}"
        )).fetchall()
    if not rows:
        print("No tasks found in DB.")
        return

    _log.info("Indexing MedQA splits to recover ground-truth letters …")
    medqa_index = _build_medqa_index()
    _log.info("Indexed %d MedQA records.", len(medqa_index))

    _log.info("Loading %d most-recent tasks from DB and rebuilding snapshots …", len(rows))
    paired: List[Tuple[Dict, AgentXAIRecord]] = []
    matched = 0
    unmatched = 0

    for (tid,) in rows:
        rec = store.get_full_record(tid)

        # Fix scoring (in-place + persisted).
        original_record = _fix_record_scoring(rec, medqa_index, store)
        if rec.input.get("patient_case", "").strip() in medqa_index:
            matched += 1
        else:
            unmatched += 1

        # Rebuild the snapshot the counterfactual engine needs.
        pipeline._snapshots[tid] = _build_snapshot_from_record(rec)
        paired.append((original_record, rec))

    _log.info("Snapshots rebuilt for %d tasks (matched=%d unmatched=%d).", len(paired), matched, unmatched)

    results = [r for _, r in paired]

    # ------------------------------------------------------------------
    # Five metrics
    # ------------------------------------------------------------------
    _log.info("Computing task performance …")
    task_perf = compute_task_performance(results)
    _log.info("  accuracy = %.1f%%  (%d/%d)", task_perf["accuracy"] * 100, task_perf["n_correct"], task_perf["n_total"])

    _log.info("Computing sufficiency (one synth re-run per task) …")
    suf = compute_sufficiency(pipeline, results)
    _log.info("  sufficiency = %.4f", suf["sufficiency_score"])

    _log.info("Computing necessity (one synth re-run per task) …")
    nec = compute_necessity(pipeline, results)
    _log.info("  necessity = %.4f", nec["necessity_score"])

    _log.info("Computing stability on %d samples (each = 1 rephrase + 1 full pipeline rerun) …", samples_for_stability)
    stab = compute_stability(pipeline, paired, n_samples=samples_for_stability)
    if stab["stability_mean_spearman"] is not None:
        _log.info("  stability rho = %.4f +/- %.4f", stab["stability_mean_spearman"], stab["stability_std_spearman"])
    else:
        _log.info("  stability ρ = N/A (no successful pairs)")

    _log.info("Computing faithfulness (one synth re-run per task) …")
    faith = compute_faithfulness(pipeline, results)
    _log.info("  faithfulness = %.4f", faith["faithfulness_score"])

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "limit": limit,
            "samples_for_stability": samples_for_stability,
            "seed": seed,
            "source": "existing_db_no_forward_rerun",
            "matched_in_medqa": matched,
            "unmatched": unmatched,
        },
        "task_performance": _safe(task_perf),
        "sufficiency":      _safe(suf),
        "necessity":        _safe(nec),
        "stability":        _safe(stab),
        "faithfulness":     _safe(faith),
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = _EVAL_DIR / f"results_existing_{timestamp}.json"
    md_path   = _EVAL_DIR / f"results_existing_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_md_report(report), encoding="utf-8")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Tasks scored:      {task_perf['n_total']}")
    print(f"  Accuracy:          {task_perf['accuracy']:.1%}  ({task_perf['n_correct']}/{task_perf['n_total']})")
    print(f"  Sufficiency:       {suf['sufficiency_score']:.1%}  ({suf['n_unchanged']}/{suf['n_total']})")
    print(f"  Necessity:         {nec['necessity_score']:.1%}  ({nec['n_changed']}/{nec['n_total']})")
    if stab["stability_mean_spearman"] is not None:
        print(f"  Stability rho:     {stab['stability_mean_spearman']:.4f} +/- {stab['stability_std_spearman']:.4f}  (n={stab['n_computed']})")
    else:
        print(f"  Stability rho:     N/A")
    print(f"  Faithfulness:      {faith['faithfulness_score']:.1%}  ({faith['n_changed']}/{faith['n_total']})")
    print()
    print(f"  Wrote: {json_path}")
    print(f"  Wrote: {md_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--samples-for-stability", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args.limit, args.samples_for_stability, args.seed)
