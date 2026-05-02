"""
Faithfulness checks for the Accountability tab.

Each check is a small pure function over the dashboard's `record` dict
(the JSON shape returned by ``GET /tasks/{task_id}``). A check returns a
``CheckResult``::

    {
        "name":        str,
        "status":      "pass" | "warn" | "fail" | "skip",
        "explanation": str,
    }

`compute_faithfulness_checks(record)` runs every check in a fixed
display order. The dashboard renders each result with a green check,
yellow warning, red cross, or gray "Not enough data" marker.

The point: the accountability report's headline numbers (top agent,
most-impactful tool, root-cause event, etc.) are *summaries* of the
underlying signals. A faithful summary should agree with what the
signals actually show. These checks are sanity assertions that flag
disagreements for the reviewer — they don't try to be ground truth.

Implemented purely on the stored record fields the API already returns;
no DB queries, no extra round-trips. If a signal is missing the check
returns ``"skip"`` rather than throwing.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List


# ---------------------------------------------------------------------------
# Constants — kept in sync with agentxai/xai/accountability.py
# ---------------------------------------------------------------------------

# Aggregator/routing actions that should not be selected as the root cause
# unless the selector explicitly fell back. See `_AGGREGATOR_ACTIONS` in
# accountability.py for the canonical list.
_AGGREGATOR_ACTIONS: frozenset = frozenset({
    "read_specialist_memories",
    "handoff_to_synthesizer",
    "decompose_case",
    "dispatch_specialists",
    "aggregate_findings",
    "compile_results",
})
_AGGREGATOR_PREFIXES: tuple = ("route_to_", "handoff_to_", "dispatch_")
_AGGREGATOR_EVENT_TYPES: frozenset = frozenset({"plan", "routing"})

# Trajectory event types that represent a tool invocation.
_TOOL_EVENT_TYPES: frozenset = frozenset({
    "tool_call", "tool_start", "tool_end", "tool_use",
})

# Match-window for "this trajectory event corresponds to this tool call".
# Tool events and the tool's own timestamp can drift slightly; 1s is well
# within typical jitter without being permissive enough to bind unrelated
# events.
_TOOL_TS_WINDOW_S: float = 1.0

# An agent counts as "high responsibility" once its score crosses this
# threshold — used by Check 6 to decide which agents to scrutinise.
# 0.35 catches the heavier side of a 2-agent split without flagging the
# loser of a clear 0.7/0.3 attribution.
_HIGH_RESPONSIBILITY: float = 0.35


# ---------------------------------------------------------------------------
# Result helper
# ---------------------------------------------------------------------------

def _result(name: str, status: str, explanation: str) -> Dict[str, Any]:
    return {"name": name, "status": status, "explanation": explanation}


def _is_aggregator(action: str, event_type: str) -> bool:
    """Mirror of `accountability._is_aggregator_node` for record dicts."""
    a = (action or "").strip().lower()
    if a in _AGGREGATOR_ACTIONS:
        return True
    if any(a.startswith(p) for p in _AGGREGATOR_PREFIXES):
        return True
    return (event_type or "").strip().lower() in _AGGREGATOR_EVENT_TYPES


def _agent_observable_signals(
    agent_id: str,
    xai: Dict[str, Any],
    report: Dict[str, Any],
) -> Dict[str, bool]:
    """
    Three observable contribution signals for one agent, derived purely
    from the stored record. The counterfactual signal is *not* directly
    observable here — it's bundled into the agent's responsibility score
    by the scorer, so a passing check on these three signals is necessary
    but not sufficient for "the responsibility number is justified".
    """
    tool_calls = xai.get("tool_calls") or []
    messages = xai.get("messages") or []
    usage = report.get("memory_usage") or []
    return {
        "non-zero tool impact": any(
            t.get("called_by") == agent_id
            and float(t.get("downstream_impact_score") or 0.0) > 0.0
            for t in tool_calls
        ),
        "acted-upon message": any(
            m.get("sender") == agent_id and bool(m.get("acted_upon"))
            for m in messages
        ),
        "cited memory": any(
            u.get("agent_id") == agent_id and bool(u.get("used_in_final_answer"))
            for u in usage
        ),
    }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_impactful_tool_on_chain(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    1. The most-impactful tool's invocation event should appear on the
       causal chain to the final outcome.

    A tool call is "on" the chain when one of the chain's events has the
    same agent_id, a tool-event type, and a timestamp within
    `_TOOL_TS_WINDOW_S` of the tool's timestamp.
    """
    name = "Most-impactful tool is on the causal path"
    xai = record.get("xai_data") or {}
    report = xai.get("accountability_report") or {}
    tool_id = report.get("most_impactful_tool_call_id") or ""
    chain = report.get("causal_chain") or []
    tool_calls = xai.get("tool_calls") or []
    trajectory = xai.get("trajectory") or []

    if not tool_id or not chain:
        return _result(name, "skip", "Not enough data — no most-impactful tool or empty causal chain.")

    tool = next((t for t in tool_calls if t.get("tool_call_id") == tool_id), None)
    if tool is None:
        return _result(
            name, "fail",
            f"Reported most_impactful_tool_call_id={tool_id[:8]} not found in tool_calls.",
        )

    chain_set = set(chain)
    called_by = tool.get("called_by") or ""
    tool_ts = float(tool.get("timestamp") or 0.0)
    for ev in trajectory:
        if ev.get("event_id") not in chain_set:
            continue
        if ev.get("agent_id") != called_by:
            continue
        if (ev.get("event_type") or "").lower() not in _TOOL_EVENT_TYPES:
            continue
        if abs(float(ev.get("timestamp") or 0.0) - tool_ts) <= _TOOL_TS_WINDOW_S:
            return _result(
                name, "pass",
                f"{tool.get('tool_name', '?')} (called by {called_by}) is on the chain.",
            )
    return _result(
        name, "warn",
        f"{tool.get('tool_name', '?')} has highest impact but no matching event "
        f"on the {len(chain)}-step causal chain.",
    )


def check_influential_message_acted_upon(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    2. The most-influential message should also be flagged ``acted_upon``.
       A message that wasn't acted upon being labelled "most influential"
       is an internal contradiction worth flagging.
    """
    name = "Most-influential message was acted upon"
    xai = record.get("xai_data") or {}
    report = xai.get("accountability_report") or {}
    msg_id = report.get("most_influential_message_id") or ""
    messages = xai.get("messages") or []

    if not msg_id:
        return _result(name, "skip", "Not enough data — no most-influential message identified.")
    msg = next((m for m in messages if m.get("message_id") == msg_id), None)
    if msg is None:
        return _result(
            name, "fail",
            "Reported message_id not found in messages list.",
        )
    if bool(msg.get("acted_upon")):
        return _result(
            name, "pass",
            f"{msg.get('sender', '?')} → {msg.get('receiver', '?')} flagged acted_upon.",
        )
    return _result(
        name, "warn",
        f"{msg.get('sender', '?')} → {msg.get('receiver', '?')} marked influential "
        "but acted_upon=false — likely a heuristic-only weight.",
    )


def check_top_agent_has_signal(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    3. The top responsible agent should have at least ONE observable
       contribution signal (impactful tool, acted-upon message, or cited
       memory). Counterfactual impact is *not* checked here because the
       stored record doesn't carry the per-agent cf delta separately —
       it's already folded into the responsibility score. This check
       therefore catches the case where the responsibility number rests
       entirely on cf / causal-centrality signals, which deserves a
       reviewer's eye.
    """
    name = "Top responsible agent has at least one observable signal"
    xai = record.get("xai_data") or {}
    report = xai.get("accountability_report") or {}
    scores = report.get("agent_responsibility_scores") or {}

    if not scores:
        return _result(name, "skip", "Not enough data — no agent_responsibility_scores recorded.")

    top_agent, top_score = max(scores.items(), key=lambda kv: float(kv[1]))
    signals = _agent_observable_signals(top_agent, xai, report)
    positive = [k for k, v in signals.items() if v]
    if positive:
        return _result(
            name, "pass",
            f"{top_agent} ({float(top_score):.2f}) has signals: " + ", ".join(positive) + ".",
        )
    return _result(
        name, "warn",
        f"{top_agent} ({float(top_score):.2f}) has no observable contribution signal "
        "(no impactful tool, no acted-upon message, no cited memory). "
        "Responsibility came from counterfactual or causal centrality only — verify.",
    )


def check_root_cause_not_aggregator(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    4. The root-cause event should not be an aggregator/routing event
       (``read_specialist_memories``, ``route_to_*``, ``handoff_to_*``,
       etc.) unless the selector explicitly fell back to one (its
       ``root_cause_reason`` will say so).
    """
    name = "Root cause is not an aggregator event"
    xai = record.get("xai_data") or {}
    report = xai.get("accountability_report") or {}
    root_id = report.get("root_cause_event_id") or ""
    reason = (report.get("root_cause_reason") or "").lower()
    trajectory = xai.get("trajectory") or []

    if not root_id:
        return _result(name, "skip", "Not enough data — no root cause identified.")
    ev = next((e for e in trajectory if e.get("event_id") == root_id), None)
    if ev is None:
        return _result(name, "fail", "Root-cause event_id not found in trajectory.")

    action = ev.get("action") or ""
    et = ev.get("event_type") or ""
    if not _is_aggregator(action, et):
        return _result(
            name, "pass",
            f"Root cause is {action or et!r} — not aggregator.",
        )
    if "no non-aggregator" in reason:
        return _result(
            name, "warn",
            f"Selector fell back to aggregator {action or et!r} because no "
            "non-aggregator ancestor was available.",
        )
    return _result(
        name, "fail",
        f"Root cause is aggregator {action or et!r} — selector should have filtered it.",
    )


def check_rationale_cites_evidence(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    5. The Synthesizer's rationale should reference retrieved evidence —
       either explicitly via ``supporting_evidence_ids`` (the new
       option-level field) or implicitly via at least one Specialist B
       memory key being marked ``used_in_final_answer`` by the
       memory-attribution heuristic.
    """
    name = "Rationale references retrieved evidence"
    system_output = record.get("system_output") or {}
    xai = record.get("xai_data") or {}
    report = xai.get("accountability_report") or {}
    rationale = (system_output.get("rationale") or "").strip()

    if not rationale:
        return _result(name, "fail", "No rationale recorded.")

    supporting = system_output.get("supporting_evidence_ids") or []
    if supporting:
        return _result(
            name, "pass",
            f"Rationale cites {len(supporting)} supporting evidence id(s).",
        )

    cited_b = [
        u for u in (report.get("memory_usage") or [])
        if u.get("agent_id") == "specialist_b" and bool(u.get("used_in_final_answer"))
    ]
    if cited_b:
        return _result(
            name, "pass",
            f"Rationale references {len(cited_b)} Specialist-B memory key(s).",
        )
    return _result(
        name, "warn",
        "Rationale recorded but no supporting_evidence_ids and no Specialist-B "
        "memory key was cited — rationale may not be grounded in retrieved evidence.",
    )


def check_no_undeserved_responsibility(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    6. No agent with responsibility >= ``_HIGH_RESPONSIBILITY`` should
       have all-zero observable signals. Catches the failure mode where
       the responsibility composite assigned a meaningful share to an
       agent whose actual contribution was empty/ignored.
    """
    name = "No high-responsibility agent with empty signals"
    xai = record.get("xai_data") or {}
    report = xai.get("accountability_report") or {}
    scores = report.get("agent_responsibility_scores") or {}

    if not scores:
        return _result(name, "skip", "Not enough data — no agent_responsibility_scores recorded.")

    flagged: List[str] = []
    for agent, score in scores.items():
        if float(score) < _HIGH_RESPONSIBILITY:
            continue
        signals = _agent_observable_signals(agent, xai, report)
        if not any(signals.values()):
            flagged.append(f"{agent}={float(score):.2f}")
    if not flagged:
        return _result(
            name, "pass",
            f"All agents with responsibility >= {_HIGH_RESPONSIBILITY:.2f} show signals.",
        )
    return _result(
        name, "warn",
        "Agent(s) with no observable contribution despite high responsibility: "
        + ", ".join(flagged) + ".",
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

# Display order in the panel.
_CHECKS: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = [
    check_impactful_tool_on_chain,
    check_influential_message_acted_upon,
    check_top_agent_has_signal,
    check_root_cause_not_aggregator,
    check_rationale_cites_evidence,
    check_no_undeserved_responsibility,
]


def compute_faithfulness_checks(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run every check; return List[CheckResult] in display order."""
    if not isinstance(record, dict):
        return []
    return [check(record) for check in _CHECKS]


def summarize_check_results(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count results by status — used by the panel header."""
    counts = {"pass": 0, "warn": 0, "fail": 0, "skip": 0}
    for r in results:
        s = r.get("status", "")
        if s in counts:
            counts[s] += 1
    return counts
