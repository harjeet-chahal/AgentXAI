"""
Attribution-quality evaluation for stored AgentXAI tasks.

Sits next to ``eval/evaluate.py`` (full pipeline re-runs) and
``eval/evaluate_existing.py`` (re-scoring sufficiency / necessity /
faithfulness over stored snapshots) — but unlike both, this script
**reads existing records only**: no LLM calls, no tool re-runs, no
specialist re-execution. It computes four quality metrics over the
accountability reports already on disk and prints a comparison summary.

The four metrics:

  1. **Empty-agent penalty** — does the report inflate the
     responsibility of agents whose observable contribution was empty
     (no impactful tools, no acted-upon messages, no cited memory)?
     Lower is better.
  2. **Impact alignment** — does the top-responsible agent also own the
     highest-impact tool or send the most-influential message? Higher
     is better.
  3. **Root-cause validity** — is the chosen root-cause event an
     ancestor of the terminal AND not a generic aggregation/routing
     event (unless the selector explicitly fell back)? Higher is better.
  4. **Faithfulness** — when the counterfactual engine zeroed each
     specialist, did the top-responsible agent's perturbation actually
     produce a larger outcome delta than a low-responsibility agent's?
     Reads the existing ``counterfactual_runs`` table; tasks that don't
     have cf rows for both agents are skipped (`n_tasks_with_data`).

Each metric is a separate function so it's easy to test in isolation
and easy to extend (a new metric is one function + one entry in
``_METRIC_RUNNERS``).

Usage::

    # Use the project's default SQLite DB.
    python -m eval.evaluate_accountability --limit 100

    # Custom DB + JSON output.
    python -m eval.evaluate_accountability \\
        --db sqlite:///path/to/agentxai.db \\
        --out-json /tmp/acc_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from agentxai.data.schemas import AgentXAIRecord
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.ui.faithfulness_checks import (
    _agent_observable_signals,
    _is_aggregator,
)


_log = logging.getLogger(__name__)


# Thresholds for "high" / "low" responsibility used by the metrics.
# Calibrated for two-specialist scoring (where a "winning" share is
# ~0.6 and a "losing" share ~0.4) but configurable from the CLI.
DEFAULT_EMPTY_RESP_THRESHOLD: float = 0.20
DEFAULT_HIGH_RESP_THRESHOLD: float = 0.50
DEFAULT_LOW_RESP_THRESHOLD: float = 0.20


# ---------------------------------------------------------------------------
# Per-task helpers — each returns a small dict; the aggregator rolls them up.
# ---------------------------------------------------------------------------

def _record_to_xai_dict(record: AgentXAIRecord) -> Dict[str, Any]:
    """
    Project a hydrated `AgentXAIRecord` to the loose dict shape the
    `faithfulness_checks` helpers expect (mirrors the API's JSON shape).
    """
    return {
        "trajectory":     [e.to_dict() for e in record.xai_data.trajectory],
        "tool_calls":     [t.to_dict() for t in record.xai_data.tool_calls],
        "messages":       [m.to_dict() for m in record.xai_data.messages],
        "memory_diffs":   [d.to_dict() for d in record.xai_data.memory_diffs],
    }


def _record_to_report_dict(record: AgentXAIRecord) -> Dict[str, Any]:
    """Project the AccountabilityReport to a plain dict."""
    if record.xai_data.accountability_report is None:
        return {}
    return record.xai_data.accountability_report.to_dict()


def _specialist_ids_from(report_dict: Dict[str, Any]) -> List[str]:
    return list(report_dict.get("agent_responsibility_scores", {}).keys())


def _top_agent(report_dict: Dict[str, Any]) -> Optional[str]:
    scores = report_dict.get("agent_responsibility_scores") or {}
    if not scores:
        return None
    return max(scores.items(), key=lambda kv: float(kv[1]))[0]


# ---------------------------------------------------------------------------
# Metric 1 — empty-agent penalty
# ---------------------------------------------------------------------------

def empty_agent_penalty_for_task(
    record: AgentXAIRecord,
    threshold: float = DEFAULT_EMPTY_RESP_THRESHOLD,
) -> Dict[str, Any]:
    """
    For each specialist on the task, mark whether they (a) have all-zero
    observable signals and (b) still received responsibility >= threshold.

    Returns ``{"task_id", "specialists": [{agent_id, responsibility,
    is_empty, is_undeserved}, ...]}``.
    """
    report = _record_to_report_dict(record)
    xai = _record_to_xai_dict(record)
    scores = report.get("agent_responsibility_scores") or {}

    specialists: List[Dict[str, Any]] = []
    for agent_id, score in scores.items():
        signals = _agent_observable_signals(agent_id, xai, report)
        is_empty = not any(signals.values())
        score_f = float(score)
        specialists.append({
            "agent_id":       agent_id,
            "responsibility": score_f,
            "is_empty":       is_empty,
            "is_undeserved":  is_empty and score_f >= threshold,
        })
    return {"task_id": record.task_id, "specialists": specialists}


def _aggregate_empty_agent(per_task: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_tasks = len(per_task)
    n_with_empty = sum(
        1 for t in per_task if any(s["is_empty"] for s in t["specialists"])
    )
    undeserved_agents = [
        s for t in per_task for s in t["specialists"] if s["is_undeserved"]
    ]
    empty_resps = [
        s["responsibility"]
        for t in per_task for s in t["specialists"] if s["is_empty"]
    ]
    return {
        "n_tasks":                  n_tasks,
        "n_tasks_with_empty_agent": n_with_empty,
        "n_undeserved_agents":      len(undeserved_agents),
        "undeserved_rate":          (
            len(undeserved_agents) / n_tasks if n_tasks else 0.0
        ),
        "mean_resp_of_empty_agents": (
            statistics.mean(empty_resps) if empty_resps else 0.0
        ),
        "max_resp_of_empty_agents": max(empty_resps) if empty_resps else 0.0,
    }


# ---------------------------------------------------------------------------
# Metric 2 — impact alignment
# ---------------------------------------------------------------------------

def impact_alignment_for_task(record: AgentXAIRecord) -> Dict[str, Any]:
    """
    Does the top responsible agent own the highest-impact tool, or send
    the most-influential message?
    """
    report = _record_to_report_dict(record)
    top_agent = _top_agent(report)

    # Highest-impact tool's caller (by downstream_impact_score).
    tools = record.xai_data.tool_calls
    if tools:
        best_tool = max(
            tools, key=lambda t: float(t.downstream_impact_score or 0.0),
        )
        # Only honor the tool when its impact is non-zero — a 0-impact
        # tool isn't really telling us anything.
        tool_owner = (
            best_tool.called_by
            if float(best_tool.downstream_impact_score or 0.0) > 0.0
            else None
        )
    else:
        tool_owner = None

    # Most-influential message's sender.
    msg_id = report.get("most_influential_message_id") or ""
    msg_sender = None
    if msg_id:
        for m in record.xai_data.messages:
            if m.message_id == msg_id:
                msg_sender = m.sender
                break

    aligned_tool = bool(top_agent and tool_owner and top_agent == tool_owner)
    aligned_msg  = bool(top_agent and msg_sender and top_agent == msg_sender)

    return {
        "task_id":                       record.task_id,
        "top_agent":                     top_agent,
        "highest_impact_tool_owner":     tool_owner,
        "most_influential_msg_sender":   msg_sender,
        "aligned_with_tool":             aligned_tool,
        "aligned_with_message":          aligned_msg,
        "aligned_either":                aligned_tool or aligned_msg,
        # `skip` signals the task had neither a non-zero tool nor a
        # message — there's no signal to align against. Excluded from the
        # alignment rate.
        "skip":                          tool_owner is None and msg_sender is None,
    }


def _aggregate_impact_alignment(per_task: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [t for t in per_task if not t["skip"]]
    n = len(scored)
    n_tool = sum(1 for t in scored if t["aligned_with_tool"])
    n_msg  = sum(1 for t in scored if t["aligned_with_message"])
    n_either = sum(1 for t in scored if t["aligned_either"])
    return {
        "n_tasks":              len(per_task),
        "n_tasks_evaluated":    n,
        "n_tasks_skipped":      len(per_task) - n,
        "n_aligned_with_tool":     n_tool,
        "n_aligned_with_message":  n_msg,
        "n_aligned_either":        n_either,
        "alignment_rate_either":   (n_either / n) if n else 0.0,
        "alignment_rate_tool":     (n_tool / n) if n else 0.0,
        "alignment_rate_message":  (n_msg / n) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Metric 3 — root-cause validity
# ---------------------------------------------------------------------------

def root_cause_validity_for_task(record: AgentXAIRecord) -> Dict[str, Any]:
    """
    Is the root-cause event id (a) on the causal chain, and (b) not an
    aggregator/routing action — unless the selector explicitly fell back?
    """
    report = _record_to_report_dict(record)
    root_id = report.get("root_cause_event_id") or ""
    chain   = report.get("causal_chain") or []
    reason  = (report.get("root_cause_reason") or "").lower()

    if not root_id:
        return {
            "task_id":   record.task_id,
            "root_cause_event_id": "",
            "in_chain":          False,
            "is_aggregator":     False,
            "fallback_marker":   False,
            "is_valid":          False,
            "skip":              True,
        }

    in_chain = root_id in set(chain)

    ev = next(
        (e for e in record.xai_data.trajectory if e.event_id == root_id),
        None,
    )
    is_aggregator = bool(ev and _is_aggregator(ev.action or "", ev.event_type or ""))
    fallback_marker = "no non-aggregator" in reason

    is_valid = in_chain and (not is_aggregator or fallback_marker)
    return {
        "task_id":             record.task_id,
        "root_cause_event_id": root_id,
        "in_chain":            in_chain,
        "is_aggregator":       is_aggregator,
        "fallback_marker":     fallback_marker,
        "is_valid":            is_valid,
        "skip":                False,
    }


def _aggregate_root_cause_validity(per_task: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [t for t in per_task if not t["skip"]]
    n = len(scored)
    n_in_chain = sum(1 for t in scored if t["in_chain"])
    n_non_aggr = sum(
        1 for t in scored
        if not t["is_aggregator"] or t["fallback_marker"]
    )
    n_valid = sum(1 for t in scored if t["is_valid"])
    return {
        "n_tasks":           len(per_task),
        "n_tasks_evaluated": n,
        "n_tasks_skipped":   len(per_task) - n,
        "n_root_in_chain":   n_in_chain,
        "n_root_non_aggregator_or_fallback": n_non_aggr,
        "n_valid":           n_valid,
        "validity_rate":     (n_valid / n) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Metric 4 — faithfulness (uses counterfactual_runs from disk)
# ---------------------------------------------------------------------------

def _agent_cf_deltas_for_task(
    store: TrajectoryStore,
    task_id: str,
) -> Dict[str, float]:
    """
    Read agent_output perturbation deltas from the counterfactual_runs
    table. Returns ``{agent_id: outcome_delta}``. Empty dict if the
    table doesn't exist or no rows match.
    """
    try:
        with store._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT target_id, outcome_delta FROM counterfactual_runs "
                    "WHERE task_id = :tid AND perturbation_type = 'agent_output'"
                ),
                {"tid": task_id},
            ).fetchall()
    except SQLAlchemyError:
        return {}
    return {r[0]: float(r[1]) for r in rows}


def faithfulness_for_task(
    record: AgentXAIRecord,
    store: TrajectoryStore,
    high_threshold: float = DEFAULT_HIGH_RESP_THRESHOLD,
    low_threshold: float = DEFAULT_LOW_RESP_THRESHOLD,
) -> Dict[str, Any]:
    """
    Compare the cf-delta of the top-responsible agent against the
    cf-delta of a low-responsibility agent. A faithful report should
    show top_delta > low_delta — zeroing the responsible agent should
    move the outcome more than zeroing the bystander.
    """
    report = _record_to_report_dict(record)
    scores = report.get("agent_responsibility_scores") or {}
    if len(scores) < 2:
        return {
            "task_id": record.task_id, "skip": True,
            "reason": "fewer than 2 specialists scored",
        }

    cf_deltas = _agent_cf_deltas_for_task(store, record.task_id)
    if len(cf_deltas) < 2:
        return {
            "task_id": record.task_id, "skip": True,
            "reason": "no agent_output cf rows for >= 2 agents",
        }

    ranked = sorted(scores.items(), key=lambda kv: float(kv[1]), reverse=True)
    top_agent, top_resp = ranked[0]
    low_agent, low_resp = ranked[-1]
    if top_agent == low_agent:
        return {"task_id": record.task_id, "skip": True,
                "reason": "single specialist"}

    if top_agent not in cf_deltas or low_agent not in cf_deltas:
        return {
            "task_id": record.task_id, "skip": True,
            "reason": "missing cf delta for top or low agent",
        }

    top_delta = cf_deltas[top_agent]
    low_delta = cf_deltas[low_agent]
    delta_gap = top_delta - low_delta

    return {
        "task_id":      record.task_id,
        "top_agent":    top_agent,
        "top_resp":     float(top_resp),
        "top_cf_delta": top_delta,
        "low_agent":    low_agent,
        "low_resp":     float(low_resp),
        "low_cf_delta": low_delta,
        "delta_gap":    delta_gap,
        "is_faithful":  delta_gap > 0.0,
        "skip":         False,
    }


def _aggregate_faithfulness(per_task: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [t for t in per_task if not t["skip"]]
    n = len(scored)
    if n == 0:
        return {
            "n_tasks":           len(per_task),
            "n_tasks_with_data": 0,
            "n_tasks_skipped":   len(per_task),
            "n_faithful":        0,
            "alignment_rate":    0.0,
            "mean_delta_gap":    0.0,
        }
    n_faithful = sum(1 for t in scored if t["is_faithful"])
    gaps = [t["delta_gap"] for t in scored]
    return {
        "n_tasks":           len(per_task),
        "n_tasks_with_data": n,
        "n_tasks_skipped":   len(per_task) - n,
        "n_faithful":        n_faithful,
        "alignment_rate":    n_faithful / n,
        "mean_delta_gap":    statistics.mean(gaps),
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def evaluate_accountability(
    store: TrajectoryStore,
    *,
    limit: Optional[int] = None,
    empty_responsibility_threshold: float = DEFAULT_EMPTY_RESP_THRESHOLD,
    high_responsibility_threshold: float = DEFAULT_HIGH_RESP_THRESHOLD,
    low_responsibility_threshold: float = DEFAULT_LOW_RESP_THRESHOLD,
) -> Dict[str, Any]:
    """
    Run all four attribution-quality metrics over the most-recent tasks
    in `store`. Returns a structured dict with per-metric aggregates plus
    per-task detail rows.

    Every task is read from the existing store — no LLM calls, no
    pipeline re-runs, no perturbations. Tasks lacking the relevant data
    for a given metric (e.g., a single-specialist task for faithfulness)
    are skipped per-metric and counted in the `n_tasks_skipped` field.
    """
    summaries = store.list_tasks(limit=limit or 50)
    task_ids = [s["task_id"] for s in summaries]

    records: List[AgentXAIRecord] = []
    for tid in task_ids:
        try:
            records.append(store.get_full_record(tid))
        except KeyError:
            _log.warning("task %s vanished between list and read; skipping", tid)
            continue

    # Restrict to tasks that actually have an accountability report; the
    # other metrics are meaningless without one.
    records = [
        r for r in records if r.xai_data.accountability_report is not None
    ]

    per_empty = [
        empty_agent_penalty_for_task(r, threshold=empty_responsibility_threshold)
        for r in records
    ]
    per_align = [impact_alignment_for_task(r) for r in records]
    per_root  = [root_cause_validity_for_task(r) for r in records]
    per_faith = [
        faithfulness_for_task(
            r, store,
            high_threshold=high_responsibility_threshold,
            low_threshold=low_responsibility_threshold,
        )
        for r in records
    ]

    return {
        "n_records_loaded": len(records),
        "thresholds": {
            "empty_responsibility": empty_responsibility_threshold,
            "high_responsibility":  high_responsibility_threshold,
            "low_responsibility":   low_responsibility_threshold,
        },
        "metrics": {
            "empty_agent_penalty": _aggregate_empty_agent(per_empty),
            "impact_alignment":    _aggregate_impact_alignment(per_align),
            "root_cause_validity": _aggregate_root_cause_validity(per_root),
            "faithfulness":        _aggregate_faithfulness(per_faith),
        },
        "per_task": {
            "empty_agent_penalty": per_empty,
            "impact_alignment":    per_align,
            "root_cause_validity": per_root,
            "faithfulness":        per_faith,
        },
    }


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def format_summary(results: Dict[str, Any]) -> str:
    """Produce a human-readable summary table for stdout / report files."""
    metrics = results.get("metrics", {})
    n_records = results.get("n_records_loaded", 0)
    thresholds = results.get("thresholds", {})

    lines: List[str] = []
    lines.append("=" * 76)
    lines.append(
        f"AgentXAI accountability evaluation — {n_records} record(s)"
    )
    lines.append(
        f"thresholds: empty>={thresholds.get('empty_responsibility', 0):.2f} · "
        f"high>={thresholds.get('high_responsibility', 0):.2f} · "
        f"low<={thresholds.get('low_responsibility', 0):.2f}"
    )
    lines.append("=" * 76)

    # 1. Empty-agent penalty.
    e = metrics.get("empty_agent_penalty", {})
    lines.append("")
    lines.append("[1] Empty-agent penalty  (lower = better)")
    lines.append(
        f"    tasks with an empty-signal specialist     : "
        f"{e.get('n_tasks_with_empty_agent', 0)} / {e.get('n_tasks', 0)}"
    )
    lines.append(
        f"    undeserved agents (empty + resp ≥ threshold): "
        f"{e.get('n_undeserved_agents', 0)} "
        f"({e.get('undeserved_rate', 0.0):.1%} of all tasks)"
    )
    lines.append(
        f"    mean responsibility for empty-signal agents : "
        f"{e.get('mean_resp_of_empty_agents', 0.0):.3f} "
        f"(max {e.get('max_resp_of_empty_agents', 0.0):.3f})"
    )

    # 2. Impact alignment.
    a = metrics.get("impact_alignment", {})
    lines.append("")
    lines.append("[2] Impact alignment  (higher = better)")
    lines.append(
        f"    tasks evaluated (had a non-zero tool or msg): "
        f"{a.get('n_tasks_evaluated', 0)} / {a.get('n_tasks', 0)} "
        f"({a.get('n_tasks_skipped', 0)} skipped)"
    )
    lines.append(
        f"    top agent owns highest-impact tool          : "
        f"{a.get('n_aligned_with_tool', 0)} ({a.get('alignment_rate_tool', 0.0):.1%})"
    )
    lines.append(
        f"    top agent sent most-influential message     : "
        f"{a.get('n_aligned_with_message', 0)} ({a.get('alignment_rate_message', 0.0):.1%})"
    )
    lines.append(
        f"    top agent aligned on EITHER signal          : "
        f"{a.get('n_aligned_either', 0)} ({a.get('alignment_rate_either', 0.0):.1%})"
    )

    # 3. Root-cause validity.
    r = metrics.get("root_cause_validity", {})
    lines.append("")
    lines.append("[3] Root-cause validity  (higher = better)")
    lines.append(
        f"    tasks with a root cause id                  : "
        f"{r.get('n_tasks_evaluated', 0)} / {r.get('n_tasks', 0)} "
        f"({r.get('n_tasks_skipped', 0)} skipped)"
    )
    lines.append(
        f"    root cause is on the causal chain           : "
        f"{r.get('n_root_in_chain', 0)}"
    )
    lines.append(
        f"    root cause is non-aggregator or marked-fb   : "
        f"{r.get('n_root_non_aggregator_or_fallback', 0)}"
    )
    lines.append(
        f"    fully valid (chain AND non-aggregator)      : "
        f"{r.get('n_valid', 0)} ({r.get('validity_rate', 0.0):.1%})"
    )

    # 4. Faithfulness.
    f = metrics.get("faithfulness", {})
    lines.append("")
    lines.append("[4] Faithfulness  (higher rate, higher gap = better)")
    lines.append(
        f"    tasks with cf rows for both top + low agent: "
        f"{f.get('n_tasks_with_data', 0)} / {f.get('n_tasks', 0)} "
        f"({f.get('n_tasks_skipped', 0)} skipped)"
    )
    lines.append(
        f"    top-agent cf-delta > low-agent cf-delta    : "
        f"{f.get('n_faithful', 0)} ({f.get('alignment_rate', 0.0):.1%})"
    )
    lines.append(
        f"    mean (top_cf_delta - low_cf_delta)         : "
        f"{f.get('mean_delta_gap', 0.0):+.3f}"
    )
    lines.append("")
    lines.append("=" * 76)
    return "\n".join(lines)


def print_summary(results: Dict[str, Any]) -> None:
    """Convenience shim — prints to stdout."""
    print(format_summary(results))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate_accountability",
        description=(
            "Compute attribution-quality metrics over stored AgentXAI "
            "task records. Reads the SQLite store directly — no LLM "
            "calls, no pipeline re-runs."
        ),
    )
    p.add_argument(
        "--db", type=str, default=None,
        help="SQLAlchemy URL for the trajectory store (default: project DB).",
    )
    p.add_argument(
        "--limit", type=int, default=50,
        help="Max number of most-recent tasks to evaluate (default: 50).",
    )
    p.add_argument(
        "--empty-threshold", type=float,
        default=DEFAULT_EMPTY_RESP_THRESHOLD,
        help=(
            "Responsibility above this counts as 'undeserved' for "
            "empty-signal agents (metric 1). Default: %(default)s."
        ),
    )
    p.add_argument(
        "--high-threshold", type=float,
        default=DEFAULT_HIGH_RESP_THRESHOLD,
        help="High-responsibility threshold for faithfulness (metric 4).",
    )
    p.add_argument(
        "--low-threshold", type=float,
        default=DEFAULT_LOW_RESP_THRESHOLD,
        help="Low-responsibility threshold for faithfulness (metric 4).",
    )
    p.add_argument(
        "--out-json", type=str, default=None,
        help="Optional path to write the full results JSON.",
    )
    p.add_argument(
        "--log-level", type=str, default="WARNING",
        help="Python logging level (default: WARNING).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper())

    store = TrajectoryStore(db_url=args.db) if args.db else TrajectoryStore()

    results = evaluate_accountability(
        store,
        limit=args.limit,
        empty_responsibility_threshold=args.empty_threshold,
        high_responsibility_threshold=args.high_threshold,
        low_responsibility_threshold=args.low_threshold,
    )
    print_summary(results)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"\nFull results written to {args.out_json}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
