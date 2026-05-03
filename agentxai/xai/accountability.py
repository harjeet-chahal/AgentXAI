"""
Pillar 7 — System-Wide Accountability.

AccountabilityReportGenerator reads every artefact produced by the other six
pillars and synthesises a single `AccountabilityReport`:

    * agent_responsibility_scores — composite per-agent score combining six
      signals (counterfactual outcome change, max tool impact, message
      efficacy, memory substance, self-reported usefulness, causal
      centrality) with weights from `_RESP_WEIGHTS`, normalized to sum to 1.
      An agent contributing nothing of substance no longer scores high
      merely because the Synthesizer's prompt structurally reads its memory.
    * root_cause_event_id        — the ancestor of the terminal (final
      diagnosis) event whose outgoing edges carry the most weight on paths
      into the terminal; ties broken by earliest timestamp.
    * causal_chain               — unweighted shortest path from root cause
      to terminal, as a list of event_ids.
    * most_impactful_tool_call_id — tool call with highest
      `downstream_impact_score`.
    * critical_memory_diffs      — diffs whose `triggered_by_event_id` lies
      on the causal_chain.
    * most_influential_message_id — message with the highest causal-edge
      weight (counterfactual_runs.outcome_delta if available, otherwise the
      acted_upon heuristic used by the DAG builder).
    * plan_deviation_summary     — one line per (agent, deviation, reason).
    * one_line_explanation       — grounded LLM sentence; falls back to a
      templated summary if no LLM is available.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from agentxai.data.schemas import (
    AccountabilityReport,
    AgentMessage,
    AgentPlan,
    AgentXAIRecord,
    CausalEdge,
    MemoryDiff,
    ToolUseEvent,
    TrajectoryEvent,
    XAIData,
)
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.causal_dag import (
    CausalDAGBuilder,
    _latest_at_or_before,
    _match_tool_event,
)
from agentxai.xai.config import DEFAULT_CONFIG, XAIScoringConfig
from agentxai.xai.counterfactual_engine import CounterfactualEngine, Pipeline
from agentxai.xai.evidence_attribution import (
    infer_supporting_evidence_ids,
    latest_top_evidence_from_memory_diffs,
    rank_most_supportive_evidence,
)
from agentxai.xai.memory_usage import (
    _influence_score,
    attribute_memory_usage,
)

_log = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-2.5-flash-lite"
_NON_SPECIALIST_AGENTS = {"orchestrator", "synthesizer", "planner", "router", ""}
_FINAL_EVENT_TYPES = {"final_diagnosis", "diagnosis_final"}


class AccountabilityReportGenerator:
    """Build, populate, and persist the capstone AccountabilityReport."""

    def __init__(
        self,
        store: TrajectoryStore,
        pipeline: Optional[Pipeline] = None,
        specialist_agents: Optional[Sequence[str]] = None,
        llm: Any = None,
        model: str = _DEFAULT_MODEL,
        config: Optional[XAIScoringConfig] = None,
    ) -> None:
        self.store = store
        self.pipeline = pipeline
        self._specialists = list(specialist_agents) if specialist_agents else None
        self.model = model
        # Scoring weights — defaults reproduce historical behaviour when
        # `config` is None. See agentxai/xai/config.py.
        self.config: XAIScoringConfig = config or DEFAULT_CONFIG

        if llm is None:
            from agentxai._llm_factory import build_gemini_llm
            llm = build_gemini_llm(model=model, temperature=0)
        self.llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        task_id: str,
        state_snapshot: Optional[Dict[str, Any]] = None,
        original_output: Optional[Dict[str, Any]] = None,
    ) -> AccountabilityReport:
        record = self.store.get_full_record(task_id)
        xai = record.xai_data

        # Pull the Synthesizer's rationale once — it drives both the
        # per-key memory_usage records and the per-agent memory_used signal.
        system_output = record.system_output or {}
        rationale = str(system_output.get("rationale", "") or "")

        # Pull the heuristic question-type label (set by the pipeline at
        # task-creation time). Used to apply per-(type, agent) priors to
        # the responsibility scores. Missing or unknown → neutral.
        task_input = record.input or {}
        question_type = str(task_input.get("question_type", "") or "unknown")

        graph = self._get_or_build_graph(record, task_id)
        terminal_id = self._find_terminal_event_id(xai)

        memory_usage = attribute_memory_usage(
            rationale=rationale,
            memory_diffs=xai.memory_diffs,
            owner_agents=self._resolve_specialists(xai) or None,
        )

        responsibility = self._responsibility_scores(
            task_id=task_id,
            xai=xai,
            state_snapshot=state_snapshot,
            original_output=original_output or dict(record.system_output),
            graph=graph,
            terminal_id=terminal_id,
            rationale=rationale,
            question_type=question_type,
        )

        root_cause_id, root_cause_reason = _select_root_cause(
            graph, terminal_id, xai, config=self.config,
        )
        chain = self._causal_chain(graph, root_cause_id, terminal_id)

        tool_id = _most_impactful_tool(xai)
        critical_diffs = _critical_memory_diffs(xai, chain)
        cf_msg_deltas = self._cf_deltas_for(task_id, "message")
        msg_id = _most_influential_message(xai, cf_msg_deltas)
        deviation_summary = _deviation_summary(xai.plans)

        final_outcome = str(system_output.get("final_diagnosis", ""))
        outcome_correct = bool(system_output.get("correct", False))

        # Evidence attribution. The Synthesizer's explicit citations live
        # on `system_output["supporting_evidence_ids"]` (the pipeline
        # already runs the heuristic fallback there). We mirror that list
        # onto the report so consumers of the accountability surface
        # don't need to also fetch system_output, AND we additionally
        # rank Specialist B's full top_evidence by support strength so
        # any high-quality docs the rationale didn't cite still surface.
        used_ids: List[str] = list(
            system_output.get("supporting_evidence_ids", []) or []
        )
        top_evidence = latest_top_evidence_from_memory_diffs(xai.memory_diffs)
        # Belt-and-suspenders: if both system_output and the inference
        # at the pipeline level missed citations (e.g., the report is
        # being regenerated from an older record), try the heuristic one
        # more time here so the report's `evidence_used_by_final_answer`
        # is non-empty when it can be.
        if not used_ids and rationale and top_evidence:
            used_ids = infer_supporting_evidence_ids(rationale, top_evidence)
        most_supportive = rank_most_supportive_evidence(top_evidence, used_ids)

        report = AccountabilityReport(
            task_id=task_id,
            final_outcome=final_outcome,
            outcome_correct=outcome_correct,
            agent_responsibility_scores=responsibility,
            root_cause_event_id=root_cause_id,
            root_cause_reason=root_cause_reason,
            causal_chain=chain,
            most_impactful_tool_call_id=tool_id,
            critical_memory_diffs=critical_diffs,
            most_influential_message_id=msg_id,
            plan_deviation_summary=deviation_summary,
            one_line_explanation="",
            memory_usage=memory_usage,
            evidence_used_by_final_answer=used_ids,
            most_supportive_evidence_ids=most_supportive,
        )
        report.one_line_explanation = self._explain(report, xai)

        self.store.save_accountability_report(report)
        return report

    # ------------------------------------------------------------------
    # Graph loading
    # ------------------------------------------------------------------

    def _get_or_build_graph(self, record: AgentXAIRecord, task_id: str) -> nx.DiGraph:
        """Use the persisted DAG if it exists; otherwise build it fresh."""
        xai = record.xai_data
        if xai.causal_graph.edges:
            return _graph_from_record(xai)
        return CausalDAGBuilder(self.store, config=self.config).build(task_id)

    # ------------------------------------------------------------------
    # Terminal / root / chain
    # ------------------------------------------------------------------

    def _find_terminal_event_id(self, xai: XAIData) -> str:
        events = xai.trajectory
        if not events:
            return ""
        for e in reversed(sorted(events, key=lambda e: e.timestamp)):
            if e.event_type in _FINAL_EVENT_TYPES or e.agent_id == "synthesizer":
                return e.event_id
        return max(events, key=lambda e: e.timestamp).event_id

    def _causal_chain(
        self, graph: nx.DiGraph, root_id: str, terminal_id: str,
    ) -> List[str]:
        if not root_id or not terminal_id:
            return []
        if root_id == terminal_id:
            return [terminal_id]
        try:
            return list(nx.shortest_path(graph, root_id, terminal_id))
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []

    # ------------------------------------------------------------------
    # Agent responsibility scoring
    # ------------------------------------------------------------------

    def _responsibility_scores(
        self,
        *,
        task_id: str,
        xai: XAIData,
        state_snapshot: Optional[Dict[str, Any]],
        original_output: Dict[str, Any],
        graph: Optional[nx.DiGraph] = None,
        terminal_id: str = "",
        rationale: str = "",
        # Defaults to "unknown" (the module-level _DEFAULT_QUESTION_TYPE
        # constant) — kept as a literal here so the class body doesn't
        # forward-reference symbols defined further down the module.
        question_type: str = "unknown",
    ) -> Dict[str, float]:
        """
        Composite per-agent responsibility scores normalized to sum to 1.

        For each detected specialist we compute six signals (see
        `_compute_responsibility_signals`) and combine them with
        `_RESP_WEIGHTS`. This avoids the failure mode where an agent with
        empty memory and an ignored message receives 0.5 responsibility purely
        because zeroing its memory happened to flip the diagnosis.

        The counterfactual outcome delta is still measured and contributes
        the largest single weight (0.35) — but it can no longer dominate the
        score on its own. An agent must additionally show real tool/message/
        memory contribution to reach a high share.

        When no pipeline is supplied, the counterfactual signal is set to 0
        for every agent and the score is computed from the remaining five
        signals; this is more informative than the previous behavior of
        distributing 1/N regardless of contribution.
        """
        specialists = self._resolve_specialists(xai)
        if not specialists:
            return {}

        cf_message_deltas = self._cf_deltas_for(task_id, "message")

        # 1. Per-agent counterfactual outcome change.
        cf_deltas: Dict[str, float] = {a: 0.0 for a in specialists}
        if self.pipeline is not None:
            engine = CounterfactualEngine(
                store=self.store,
                pipeline=self.pipeline,
                task_id=task_id,
                state_snapshot=state_snapshot,
                original_output=original_output,
                config=self.config,
            )
            for agent in specialists:
                try:
                    cf_deltas[agent] = float(engine.perturb_agent_output(agent))
                except Exception as exc:  # pragma: no cover - defensive
                    _log.warning(
                        "Responsibility perturbation failed for %s: %s", agent, exc,
                    )
                    cf_deltas[agent] = 0.0

        # 2. Bundle the six signals per agent.
        raw_signals: Dict[str, Dict[str, float]] = {
            agent: _compute_responsibility_signals(
                agent,
                xai=xai,
                cf_outcome_delta=cf_deltas[agent],
                cf_message_deltas=cf_message_deltas,
                graph=graph,
                terminal_id=terminal_id,
                rationale=rationale,
            )
            for agent in specialists
        }

        # 3. Max-normalize causal_centrality across agents into [0, 1] so it
        #    composes cleanly with the other (already in-range) signals.
        max_cc = max(
            (s["causal_centrality"] for s in raw_signals.values()),
            default=0.0,
        )
        if max_cc > 0:
            for s in raw_signals.values():
                s["causal_centrality"] = s["causal_centrality"] / max_cc

        combined = {
            agent: _combine_signals(s, weights=self.config.responsibility_weights)
            for agent, s in raw_signals.items()
        }

        # Apply per-(question_type, agent) priors before normalising. With
        # the default `question_type="unknown"`, every prior is 1.0 and
        # the scores pass through unchanged — preserving every existing
        # test and stored record's ordering exactly.
        if question_type and question_type != "unknown":
            combined = {
                agent: combined[agent] * _prior_for(
                    question_type, agent,
                    priors=self.config.question_type_priors,
                )
                for agent in combined
            }

        return _normalize_to_one(combined)

    def _resolve_specialists(self, xai: XAIData) -> List[str]:
        if self._specialists is not None:
            return list(self._specialists)
        seen: List[str] = []
        for e in xai.trajectory:
            if e.agent_id in _NON_SPECIALIST_AGENTS:
                continue
            if e.agent_id not in seen:
                seen.append(e.agent_id)
        return seen

    # ------------------------------------------------------------------
    # Counterfactual-run lookups
    # ------------------------------------------------------------------

    def _cf_deltas_for(self, task_id: str, perturbation_type: str) -> Dict[str, float]:
        try:
            with self.store._engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT target_id, outcome_delta FROM counterfactual_runs "
                        "WHERE task_id = :tid AND perturbation_type = :p"
                    ),
                    {"tid": task_id, "p": perturbation_type},
                ).fetchall()
        except SQLAlchemyError:
            return {}
        return {r[0]: float(r[1]) for r in rows}

    # ------------------------------------------------------------------
    # LLM explanation
    # ------------------------------------------------------------------

    def _explain(self, report: AccountabilityReport, xai: XAIData) -> str:
        if self.llm is None:
            return _fallback_explanation(report, xai, config=self.config)
        prompt = _build_explanation_prompt(report, xai, config=self.config)
        try:
            response = self.llm.invoke(prompt)
            text_out = _extract_text(response).strip()
        except Exception:
            return _fallback_explanation(report, xai, config=self.config)
        return text_out or _fallback_explanation(report, xai, config=self.config)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _graph_from_record(xai: XAIData) -> nx.DiGraph:
    g: nx.DiGraph = nx.DiGraph()
    for e in xai.trajectory:
        g.add_node(
            e.event_id,
            agent_id=e.agent_id,
            event_type=e.event_type,
            action=e.action,
            timestamp=e.timestamp,
        )
    for edge in xai.causal_graph.edges:
        g.add_edge(
            edge.cause_event_id,
            edge.effect_event_id,
            edge_id=edge.edge_id,
            weight=float(edge.causal_strength),
            causal_type=edge.causal_type,
        )
    return g


def _most_impactful_tool(xai: XAIData) -> str:
    """
    Return the tool_call_id with the highest counterfactual impact score.

    Even when every tool call yields a 0.0 delta — common with deterministic
    LLM specialists that are robust to single-tool ablation — we still return
    the highest-scored call so the dashboard surfaces a representative tool
    rather than "Not recorded". The score itself stays honest (0.00), so the
    UI / explanation can flag the no-measurable-impact case explicitly.
    """
    if not xai.tool_calls:
        return ""
    best = max(xai.tool_calls, key=lambda t: float(t.downstream_impact_score or 0.0))
    return best.tool_call_id


def _critical_memory_diffs(xai: XAIData, chain: Sequence[str]) -> List[str]:
    if not chain:
        return []
    chain_set = set(chain)
    return [d.diff_id for d in xai.memory_diffs if d.triggered_by_event_id in chain_set]


def _most_influential_message(xai: XAIData, cf_deltas: Dict[str, float]) -> str:
    if not xai.messages:
        return ""
    best_id, best_w = "", float("-inf")
    for m in xai.messages:
        w = cf_deltas.get(m.message_id)
        if w is None:
            w = 1.0 if m.acted_upon else 0.5
        if w > best_w:
            best_w, best_id = w, m.message_id
    return best_id


def _deviation_summary(plans: Iterable[AgentPlan]) -> str:
    lines: List[str] = []
    for p in plans:
        reasons = list(p.deviation_reasons) + [""] * max(
            0, len(p.deviations) - len(p.deviation_reasons)
        )
        for dev, reason in zip(p.deviations, reasons):
            lines.append(f"{p.agent_id}: {dev} — {reason}".rstrip(" —"))
    return "\n".join(lines)


def _normalize_to_one(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, s) for s in scores.values())
    if total <= 0.0:
        if not scores:
            return {}
        share = 1.0 / len(scores)
        return {a: share for a in scores}
    return {a: max(0.0, s) / total for a, s in scores.items()}


# ---------------------------------------------------------------------------
# Per-agent responsibility signals
# ---------------------------------------------------------------------------
#
# Each signal returns a value in [0, 1] capturing one facet of an agent's
# real contribution to the final outcome. They are combined by
# `_combine_signals` using `_RESP_WEIGHTS`. The motivation: relying on the
# counterfactual outcome delta alone gives high responsibility to agents
# whose memory was *touched* by the Synthesizer but carried no actual
# information, because the perturbation flips the diagnosis through prompt
# structure / LLM nondeterminism rather than through lost content.

# Backward-compat alias. Source of truth lives on
# `XAIScoringConfig.responsibility_weights` — see agentxai/xai/config.py.
# Held as a frozen copy so external code that mutates this dict (tests
# included) doesn't accidentally tamper with DEFAULT_CONFIG.
_RESP_WEIGHTS: Dict[str, float] = dict(DEFAULT_CONFIG.responsibility_weights)


# ---------------------------------------------------------------------------
# Per-(question_type, agent) responsibility priors
# ---------------------------------------------------------------------------
#
# A question's type (set by `agentxai.data.question_classifier`) tells us
# how much weight each agent's findings *should* deserve before we even
# look at the run-time signals. Specialist A (symptom analyzer) is the
# right specialist for a classic diagnosis question, but for a
# screening_or_test or pharmacology question the symptom analysis is
# largely irrelevant — Specialist B (evidence retriever / guidelines) is
# doing the actual work. Multiplying the composite score by the prior
# and renormalising shrinks A's share when its work doesn't match the
# question type.
#
# Priors are deliberately gentle (0.5–1.0) so the run-time signals still
# dominate when the agent did do meaningful work; the prior only breaks
# near-ties in the right direction.

_DEFAULT_QUESTION_TYPE: str = "unknown"

# Backward-compat alias. Source of truth lives on
# `XAIScoringConfig.question_type_priors` — see agentxai/xai/config.py.
# Frozen copy so external mutation can't poison DEFAULT_CONFIG.
_QUESTION_TYPE_PRIORS: Dict[str, Dict[str, float]] = {
    qt: dict(agents)
    for qt, agents in DEFAULT_CONFIG.question_type_priors.items()
}


def _prior_for(
    question_type: str,
    agent_id: str,
    priors: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """
    Lookup the multiplicative (question_type, agent) prior.

    Returns 1.0 (neutral) when the pair isn't tabulated. Pass an
    instance-level `priors` dict (typically from
    ``XAIScoringConfig.question_type_priors``) to ablate.
    """
    table_set = priors if priors is not None else DEFAULT_CONFIG.question_type_priors
    table = table_set.get(question_type) or {}
    return float(table.get(agent_id, 1.0))


def _is_substantive(value: Any) -> bool:
    """
    Return True iff a memory value carries non-trivial information.

    Empty containers, blank strings, None, False, and numeric zero are all
    considered uninformative. This is the gating predicate used by
    `_agent_memory_substance`.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, (str, bytes, dict, list, tuple, set, frozenset)):
        return len(value) > 0
    return True


def _agent_tool_impact(agent_id: str, tool_calls: Iterable[ToolUseEvent]) -> float:
    """
    Strongest counterfactual tool impact attributable to `agent_id`.

    Uses max (not mean) so a single high-impact tool is not diluted by a
    handful of low-impact ones from the same agent. Returns 0 if the agent
    invoked no tools.
    """
    own = [tc for tc in tool_calls if tc.called_by == agent_id]
    if not own:
        return 0.0
    return max(
        max(0.0, min(1.0, float(tc.downstream_impact_score or 0.0)))
        for tc in own
    )


def _agent_message_efficacy(
    agent_id: str,
    messages: Iterable[AgentMessage],
    cf_message_deltas: Dict[str, float],
) -> float:
    """
    Strongest evidence that one of `agent_id`'s outgoing messages mattered.

    For each message the agent sent, prefer its logged counterfactual delta
    if available; otherwise fall back to a heuristic (1.0 if marked
    `acted_upon`, else 0.2). Returns 0 if the agent sent no messages — and
    a low value if every message it sent was ignored.
    """
    own = [m for m in messages if m.sender == agent_id]
    if not own:
        return 0.0
    weights: List[float] = []
    for m in own:
        cf = cf_message_deltas.get(m.message_id)
        if cf is not None:
            weights.append(max(0.0, min(1.0, float(cf))))
        else:
            weights.append(1.0 if m.acted_upon else 0.2)
    return max(weights)


def _agent_memory_substance(
    agent_id: str,
    memory_diffs: Iterable[MemoryDiff],
) -> float:
    """
    Fraction of `agent_id`'s memory *writes* whose value_after is substantive.

    This is the structural fallback used when no rationale text is available
    (e.g., the Synthesizer failed and produced no rationale to scan). Agents
    that wrote only `confidence=0` / `top_conditions=[]` score 0 here even
    when the Synthesizer technically read those keys.
    """
    writes = [
        d for d in memory_diffs
        if d.agent_id == agent_id and d.operation == "write"
    ]
    if not writes:
        return 0.0
    n_substantive = sum(1 for d in writes if _is_substantive(d.value_after))
    return n_substantive / len(writes)


def _agent_memory_used(
    agent_id: str,
    memory_diffs: Iterable[MemoryDiff],
    rationale: str,
) -> float:
    """
    Fraction of `agent_id`'s substantive memory writes that were *cited*
    in the Synthesizer's rationale.

    Uses the same substring heuristic as
    `agentxai.xai.memory_usage.attribute_memory_usage` so the per-key
    detail surfaced on the report and the per-agent score stay consistent.
    Falls back to `_agent_memory_substance` when the rationale is empty.

    The point: distinguish "the Synthesizer's prompt loaded this memory"
    (cheap and uniform) from "this memory's content actually shows up in
    the conclusion" (the signal we actually care about for blame
    assignment).
    """
    if not rationale:
        return _agent_memory_substance(agent_id, memory_diffs)

    # Take the latest write per key for this agent, dropping
    # uninformative values up front so `confidence=0` / `top_conditions=[]`
    # never enter the denominator.
    latest_substantive: Dict[str, Any] = {}
    sorted_diffs = sorted(
        (
            d for d in memory_diffs
            if d.agent_id == agent_id and d.operation == "write" and d.key
        ),
        key=lambda d: d.timestamp,
    )
    for d in sorted_diffs:
        if _is_substantive(d.value_after):
            latest_substantive[d.key] = d.value_after
        else:
            # A later empty write erases the prior substantive value.
            latest_substantive.pop(d.key, None)

    if not latest_substantive:
        return 0.0

    rationale_lower = rationale.lower()
    scores = [_influence_score(v, rationale_lower) for v in latest_substantive.values()]
    return sum(scores) / len(scores)


def _final_memory_state(
    agent_id: str,
    memory_diffs: Iterable[MemoryDiff],
) -> Dict[str, Any]:
    """Replay an agent's write diffs in timestamp order to recover its final memory."""
    state: Dict[str, Any] = {}
    own_writes = sorted(
        (d for d in memory_diffs if d.agent_id == agent_id and d.operation == "write"),
        key=lambda d: d.timestamp,
    )
    for d in own_writes:
        state[d.key] = d.value_after
    return state


_USEFULNESS_KEYS = ("confidence", "retrieval_confidence", "severity_score")


def _agent_usefulness(
    agent_id: str,
    memory_diffs: Iterable[MemoryDiff],
) -> float:
    """
    Agent's self-reported usefulness as the largest of its `confidence`-like
    final-memory keys, clamped to [0, 1]. Returns 0 if no such key was written.

    The standard specialist contract emits `confidence` (SpecialistA),
    `retrieval_confidence` (SpecialistB), or `severity_score` — these are the
    fields we trust as a coarse self-report.
    """
    final = _final_memory_state(agent_id, memory_diffs)
    best = 0.0
    for key in _USEFULNESS_KEYS:
        if key not in final:
            continue
        try:
            best = max(best, max(0.0, min(1.0, float(final[key]))))
        except (TypeError, ValueError):
            continue
    return best


def _agent_causal_centrality(
    agent_id: str,
    graph: Optional["nx.DiGraph"],
    terminal_id: str,
) -> float:
    """
    Raw outgoing edge weight from `agent_id`'s events into the terminal's
    reachable set. Caller must max-normalize across agents to map this into
    [0, 1] — returning the raw weight here keeps the relative ordering exact
    when the graph is dominated by a single agent.

    Returns 0 when the graph is missing, the terminal has no ancestors, or
    none of the agent's events reach the terminal.
    """
    if graph is None or not terminal_id or terminal_id not in graph:
        return 0.0
    try:
        ancestors = nx.ancestors(graph, terminal_id)
    except (nx.NodeNotFound, nx.NetworkXError):
        return 0.0
    if not ancestors:
        return 0.0

    total = 0.0
    for node in ancestors:
        if graph.nodes[node].get("agent_id") != agent_id:
            continue
        for v in graph.successors(node):
            if v == terminal_id or v in ancestors:
                total += float(graph[node][v].get("weight", 0.0))
    return total


def _compute_responsibility_signals(
    agent_id: str,
    *,
    xai: XAIData,
    cf_outcome_delta: float,
    cf_message_deltas: Dict[str, float],
    graph: Optional["nx.DiGraph"],
    terminal_id: str,
    rationale: str = "",
) -> Dict[str, float]:
    """
    Bundle all six per-agent responsibility signals into a single dict.

    `causal_centrality` is left as a *raw* outgoing-weight here; the caller
    is expected to max-normalize it across all agents before combining (see
    `_responsibility_scores`). Every other signal is already in [0, 1].

    `rationale` (the Synthesizer's natural-language explanation, available
    in `system_output["rationale"]`) drives the `memory_used` signal.
    Pass "" to fall back to the looser `memory_substance` proxy.
    """
    return {
        "counterfactual":   max(0.0, min(1.0, float(cf_outcome_delta))),
        "tool_impact":      _agent_tool_impact(agent_id, xai.tool_calls),
        "message_efficacy": _agent_message_efficacy(
            agent_id, xai.messages, cf_message_deltas
        ),
        "memory_used":      _agent_memory_used(
            agent_id, xai.memory_diffs, rationale,
        ),
        "usefulness":       _agent_usefulness(agent_id, xai.memory_diffs),
        "causal_centrality": _agent_causal_centrality(agent_id, graph, terminal_id),
    }


def _combine_signals(
    signals: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Weighted sum of per-signal scores. Result in [0, 1] when each signal
    is in [0, 1] and `weights.values()` sums to ~1.0.

    If `weights` is None, falls back to the historical defaults
    (`DEFAULT_CONFIG.responsibility_weights`). Pass an instance-level
    weight map (e.g., from a `XAIScoringConfig`) to ablate.
    """
    w = weights if weights is not None else DEFAULT_CONFIG.responsibility_weights
    return sum(
        weight * float(signals.get(name, 0.0))
        for name, weight in w.items()
    )


# ---------------------------------------------------------------------------
# Root-cause selection
# ---------------------------------------------------------------------------
#
# The previous selector picked the ancestor with the highest outgoing graph
# weight to the terminal-reachable set, breaking ties by earliest timestamp.
# That awarded the title to late aggregator events like
# `read_specialist_memories` (the Synthesizer's own pre-decision step) — they
# sit one hop from the terminal with a strong direct edge, so they trivially
# win the weight contest even though they only *forward* upstream signal.
#
# The new selector restricts candidates to non-aggregator ancestors and
# combines four signals: outgoing graph weight, the impact of any tool the
# event invoked, the substantiveness of memory writes it triggered, and
# whether it produced acted-upon outgoing messages. A mild upstream factor
# breaks near-ties in favor of earlier events.

# Action names that only forward / aggregate / route signal — never the
# meaningful root cause. Match is case-insensitive and exact.
_AGGREGATOR_ACTIONS: frozenset = frozenset({
    "read_specialist_memories",
    "handoff_to_synthesizer",
    "decompose_case",
    "dispatch_specialists",
    "aggregate_findings",
    "compile_results",
})

# Action prefixes that mark routing/dispatch operations regardless of
# specific destination ("route_to_specialist_a", "handoff_to_x", etc.).
_AGGREGATOR_ACTION_PREFIXES: Tuple[str, ...] = (
    "route_to_",
    "handoff_to_",
    "dispatch_",
)

# Non-decision event types that should never be the root cause on their own.
_AGGREGATOR_EVENT_TYPES: frozenset = frozenset({"plan", "routing"})

# Backward-compat aliases. Source of truth lives on `XAIScoringConfig`
# — see agentxai/xai/config.py. Tool impact is the strongest single
# signal because high-impact tools are almost always the meaningful
# upstream cause; messages and memory writes are smaller boosters.
_ROOT_TOOL_BONUS: float = DEFAULT_CONFIG.root_tool_bonus
_ROOT_ACTED_MSG_BONUS: float = DEFAULT_CONFIG.root_acted_msg_bonus
_ROOT_MEM_BONUS: float = DEFAULT_CONFIG.root_mem_bonus
_ROOT_UPSTREAM_DISCOUNT: float = DEFAULT_CONFIG.root_upstream_discount


def _is_aggregator_node(node_attrs: Dict[str, Any]) -> bool:
    """
    True if a graph node represents a routing / aggregating action that
    should not be selected as the root cause on its own.
    """
    action = (node_attrs.get("action") or "").strip().lower()
    if action in _AGGREGATOR_ACTIONS:
        return True
    if any(action.startswith(p) for p in _AGGREGATOR_ACTION_PREFIXES):
        return True
    event_type = (node_attrs.get("event_type") or "").strip().lower()
    if event_type in _AGGREGATOR_EVENT_TYPES:
        return True
    return False


def _events_by_agent(
    trajectory: Iterable[TrajectoryEvent],
) -> Dict[str, List[TrajectoryEvent]]:
    """Group trajectory events by `agent_id`, sorted by timestamp within each agent."""
    out: Dict[str, List[TrajectoryEvent]] = defaultdict(list)
    for e in trajectory:
        out[e.agent_id].append(e)
    for evs in out.values():
        evs.sort(key=lambda e: e.timestamp)
    return out


def _event_tool_impact_index(
    xai: XAIData,
    by_agent: Dict[str, List[TrajectoryEvent]],
) -> Dict[str, float]:
    """
    Map each trajectory event_id to the best `downstream_impact_score` of any
    tool call associated with that event (matched via the same heuristic
    used by the causal-DAG builder, so the indexes stay consistent).
    """
    out: Dict[str, float] = {}
    for tc in xai.tool_calls:
        ev = _match_tool_event(by_agent.get(tc.called_by, []), tc)
        if ev is None:
            continue
        score = max(0.0, min(1.0, float(tc.downstream_impact_score or 0.0)))
        if score > out.get(ev.event_id, 0.0):
            out[ev.event_id] = score
    return out


def _event_substantive_memory_index(xai: XAIData) -> Dict[str, int]:
    """
    Map each trajectory event_id to the count of substantive memory writes
    it triggered (where `_is_substantive(value_after)` holds).
    """
    out: Dict[str, int] = defaultdict(int)
    for d in xai.memory_diffs:
        if d.operation != "write" or not d.triggered_by_event_id:
            continue
        if _is_substantive(d.value_after):
            out[d.triggered_by_event_id] += 1
    return dict(out)


def _event_acted_message_index(
    xai: XAIData,
    by_agent: Dict[str, List[TrajectoryEvent]],
) -> Dict[str, int]:
    """
    Map each trajectory event_id to the count of acted-upon outgoing
    messages the agent sent at-or-after that event (using the same
    "latest event at-or-before message" anchor as the causal-DAG builder).
    """
    out: Dict[str, int] = defaultdict(int)
    for m in xai.messages:
        if not m.acted_upon or not m.sender:
            continue
        src = _latest_at_or_before(by_agent.get(m.sender, []), m.timestamp)
        if src is None:
            continue
        out[src.event_id] += 1
    return dict(out)


def _root_cause_base_weight(
    event_id: str,
    graph: "nx.DiGraph",
    reachable: set,
) -> float:
    """Sum of outgoing edge weights from `event_id` into the terminal-reachable set."""
    total = 0.0
    for v in graph.successors(event_id):
        if v in reachable:
            total += float(graph[event_id][v].get("weight", 0.0))
    return total


def _root_cause_score(
    event_id: str,
    *,
    graph: "nx.DiGraph",
    reachable: set,
    tool_impact: Dict[str, float],
    mem_writes: Dict[str, int],
    acted_msgs: Dict[str, int],
    ts_min: float,
    ts_max: float,
    config: Optional[XAIScoringConfig] = None,
) -> Tuple[float, List[str]]:
    """
    Score one candidate event and return (score, reason_fragments).

    The score has three layers:
        raw    = base graph weight + tool/message/memory bonuses
        scaled = raw * upstream_factor   (later events get up to
                  `config.root_upstream_discount` shaved off)
    The reason fragments describe *which* bonuses fired, in plain English.

    Pass a custom :class:`XAIScoringConfig` to override the bonus weights.
    """
    cfg = config or DEFAULT_CONFIG
    base = _root_cause_base_weight(event_id, graph, reachable)
    reasons: List[str] = []
    bonus = 0.0

    ti = tool_impact.get(event_id, 0.0)
    if ti > 0.0:
        bonus += cfg.root_tool_bonus * ti
        reasons.append(f"high-impact tool ({ti:.2f})")

    am = acted_msgs.get(event_id, 0)
    if am > 0:
        bonus += cfg.root_acted_msg_bonus * am
        reasons.append(
            "acted-upon message" + ("s" if am > 1 else "")
        )

    mw = mem_writes.get(event_id, 0)
    if mw > 0:
        bonus += cfg.root_mem_bonus * mw
        reasons.append(
            "substantive memory write" + ("s" if mw > 1 else "")
        )

    raw = base + bonus
    ts = float(graph.nodes[event_id].get("timestamp", ts_min))
    if ts_max > ts_min:
        ts_norm = (ts - ts_min) / (ts_max - ts_min)
    else:
        ts_norm = 0.0
    upstream_factor = 1.0 - cfg.root_upstream_discount * ts_norm

    return raw * upstream_factor, reasons


def _format_root_cause_reason(
    event_id: str,
    graph: "nx.DiGraph",
    reasons: List[str],
) -> str:
    """Compose a one-line, human-readable explanation for the chosen root cause."""
    attrs = graph.nodes.get(event_id, {})
    label = (
        (attrs.get("action") or "").strip()
        or (attrs.get("event_type") or "").strip()
        or "event"
    )
    agent = attrs.get("agent_id") or "unknown"
    if reasons:
        return f"{label} from {agent}: " + ", ".join(reasons)
    return f"{label} from {agent} carried highest causal weight to the final outcome"


def _select_root_cause(
    graph: "nx.DiGraph",
    terminal_id: str,
    xai: XAIData,
    config: Optional[XAIScoringConfig] = None,
) -> Tuple[str, str]:
    """
    Pick the most explanatory root-cause event among ancestors of `terminal_id`.

    Algorithm:
      1. Restrict to ancestors of the terminal in the causal DAG.
      2. Drop pure aggregator/routing actions (see `_is_aggregator_node`).
         If that empties the candidate pool — pathological run with only
         routing events — fall back to the unfiltered ancestor set so we
         still produce *something* rather than empty.
      3. Score each remaining candidate via `_root_cause_score`, which
         combines outgoing graph weight with tool/message/memory bonuses
         and a mild upstream-preference factor.
      4. Pick the highest score; tie-break by earliest timestamp.
      5. Return (event_id, human-readable reason).

    Returns ("", "") when no terminal or no ancestors exist.

    Pass a custom :class:`XAIScoringConfig` to override the bonus weights
    used by ``_root_cause_score``; aggregator filtering is unaffected.
    """
    cfg = config or DEFAULT_CONFIG
    if not terminal_id or terminal_id not in graph:
        return "", ""

    ancestors = nx.ancestors(graph, terminal_id)
    if not ancestors:
        return "", ""

    reachable = ancestors | {terminal_id}

    candidates = [a for a in ancestors if not _is_aggregator_node(graph.nodes[a])]
    used_fallback = False
    if not candidates:
        candidates = list(ancestors)
        used_fallback = True

    by_agent = _events_by_agent(xai.trajectory)
    tool_impact = _event_tool_impact_index(xai, by_agent)
    mem_writes  = _event_substantive_memory_index(xai)
    acted_msgs  = _event_acted_message_index(xai, by_agent)

    timestamps = [
        float(graph.nodes[c].get("timestamp", 0.0)) for c in candidates
    ]
    ts_min = min(timestamps) if timestamps else 0.0
    ts_max = max(timestamps) if timestamps else 0.0

    scored: List[Tuple[float, float, str, List[str]]] = []
    for c in candidates:
        score, reasons = _root_cause_score(
            c,
            graph=graph, reachable=reachable,
            tool_impact=tool_impact,
            mem_writes=mem_writes,
            acted_msgs=acted_msgs,
            ts_min=ts_min, ts_max=ts_max,
            config=cfg,
        )
        ts = float(graph.nodes[c].get("timestamp", float("inf")))
        # We sort descending by (score, -timestamp) — higher score wins,
        # earlier timestamp breaks ties.
        scored.append((score, -ts, c, reasons))

    best_score, _, best_id, best_reasons = max(scored, key=lambda t: (t[0], t[1]))
    reason = _format_root_cause_reason(best_id, graph, best_reasons)
    if used_fallback:
        reason += " (no non-aggregator ancestor; selected from full ancestor set)"
    return best_id, reason


# ---------------------------------------------------------------------------
# UUID-to-entity resolvers (used by both LLM prompt and fallback)
# ---------------------------------------------------------------------------

def _resolve_tool_call(xai: XAIData, tool_call_id: str) -> Optional[Dict[str, Any]]:
    if not tool_call_id:
        return None
    for tc in xai.tool_calls:
        if tc.tool_call_id == tool_call_id:
            return {
                "tool_name": tc.tool_name,
                "called_by": tc.called_by,
                "impact_score": round(float(tc.downstream_impact_score or 0.0), 2),
            }
    return None


def _resolve_event(xai: XAIData, event_id: str) -> Optional[Dict[str, Any]]:
    if not event_id:
        return None
    for e in xai.trajectory:
        if e.event_id == event_id:
            return {
                "event_type": e.event_type,
                "agent_id": e.agent_id,
                "action": e.action or "",
            }
    return None


def _resolve_message(xai: XAIData, message_id: str) -> Optional[Dict[str, Any]]:
    if not message_id:
        return None
    for m in xai.messages:
        if m.message_id == message_id:
            return {
                "sender": m.sender,
                "receiver": m.receiver,
                "message_type": m.message_type,
                "acted_upon": bool(m.acted_upon),
            }
    return None


# ---------------------------------------------------------------------------
# LLM prompting
# ---------------------------------------------------------------------------

# Backward-compat alias. Source of truth: `XAIScoringConfig.tie_epsilon`.
_TIE_EPSILON: float = DEFAULT_CONFIG.tie_epsilon


def _tied_top_agents(
    scores: Dict[str, float],
    epsilon: Optional[float] = None,
) -> List[str]:
    """Return all agents within `epsilon` of the top score, ranked desc.

    `epsilon` defaults to ``DEFAULT_CONFIG.tie_epsilon``; pass an
    instance-level value (typically ``self.config.tie_epsilon``) to ablate.
    """
    if not scores:
        return []
    eps = epsilon if epsilon is not None else DEFAULT_CONFIG.tie_epsilon
    ranked = sorted(scores.items(), key=lambda kv: float(kv[1]), reverse=True)
    top = float(ranked[0][1])
    return [a for a, s in ranked if top - float(s) < eps]


def _build_explanation_prompt(
    report: AccountabilityReport,
    xai: XAIData,
    config: Optional[XAIScoringConfig] = None,
) -> str:
    cfg = config or DEFAULT_CONFIG
    # Resolve raw UUIDs into descriptive dicts so the LLM never has to
    # reference an opaque id in its sentence.
    tool = _resolve_tool_call(xai, report.most_impactful_tool_call_id)
    # Suppress the tool field when its measured impact is effectively zero —
    # otherwise the LLM is invited to call a no-op tool "impactful".
    if tool and float(tool.get("impact_score") or 0.0) <= 0.0:
        tool = None

    scores = report.agent_responsibility_scores or {}
    tied = _tied_top_agents(scores, epsilon=cfg.tie_epsilon)

    fields: Dict[str, Any] = {
        "final_outcome": report.final_outcome,
        "outcome_correct": report.outcome_correct,
        "agent_responsibility_scores": {
            agent: round(float(score), 2)
            for agent, score in scores.items()
        },
        # Pre-computed: agents within 0.05 of the top responsibility score.
        # If this list has more than one entry, the responsibility split is
        # essentially a tie — the explanation must acknowledge all of them.
        "tied_top_agents": tied,
        "root_cause_event": _resolve_event(xai, report.root_cause_event_id),
        # One-line rationale for *why* the selector picked that root-cause
        # event (e.g., "textbook_search from specialist_b: high-impact tool").
        # The LLM should weave this into the explanation rather than restating
        # the event_type in isolation.
        "root_cause_reason": report.root_cause_reason or "",
        "most_impactful_tool_call": tool,
        "most_influential_message": _resolve_message(
            xai, report.most_influential_message_id
        ),
        "causal_chain_length": len(report.causal_chain),
        "critical_memory_diff_count": len(report.critical_memory_diffs),
        "plan_deviation_summary": report.plan_deviation_summary,
    }
    return (
        "You are writing the one-sentence accountability summary for a "
        "multi-agent medical-triage run. Use ONLY the structured fields below "
        "— do not invent any entities, agents, tools, or diagnoses that are "
        "not named there.\n\n"
        "FORMAT RULES:\n"
        "  * Refer to entities by their human-readable fields only "
        "(tool_name, agent_id, event_type, sender→receiver). NEVER quote "
        "raw UUIDs, event_ids, message_ids, or tool_call_ids.\n"
        "  * If most_impactful_tool_call is null, do NOT claim any tool was "
        "impactful — focus the sentence on the responsible agent and the "
        "most-influential message instead.\n"
        "  * If tied_top_agents has more than one entry, do NOT single out "
        "any one of them as 'primarily responsible' — instead say they "
        "contributed equally (or jointly drove the outcome). Single out one "
        "agent only when tied_top_agents has exactly one entry.\n"
        "  * One plain English sentence, no markdown, no preamble.\n\n"
        f"{json.dumps(fields, indent=2, default=str)}"
    )


def _extract_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _fallback_explanation(
    report: AccountabilityReport,
    xai: Optional[XAIData] = None,
    config: Optional[XAIScoringConfig] = None,
) -> str:
    cfg = config or DEFAULT_CONFIG
    correct = "correct" if report.outcome_correct else "incorrect"
    parts: List[str] = [f"Diagnosis {report.final_outcome!r} was {correct}"]

    scores = report.agent_responsibility_scores or {}
    tied = _tied_top_agents(scores, epsilon=cfg.tie_epsilon)
    if len(tied) > 1:
        # Tie — credit the whole group rather than picking one arbitrarily.
        top_score = float(scores[tied[0]])
        joined = ", ".join(tied[:-1]) + " and " + tied[-1]
        parts.append(
            f"responsibility shared across {joined} "
            f"({top_score:.2f} each)"
        )
    elif tied:
        agent = tied[0]
        parts.append(
            f"largest responsibility on {agent} ({float(scores[agent]):.2f})"
        )

    if xai is not None:
        tool = _resolve_tool_call(xai, report.most_impactful_tool_call_id)
        # Only mention the tool when its measured impact is non-zero —
        # otherwise the sentence falsely implies the tool drove the outcome.
        if tool and float(tool.get("impact_score") or 0.0) > 0.0:
            parts.append(
                f"most-impactful tool was {tool['tool_name']} "
                f"called by {tool['called_by']} "
                f"(impact {tool['impact_score']:.2f})"
            )
        # Prefer the rich `root_cause_reason` produced by the selector — it
        # already names the event/agent and the bonuses that fired. Only
        # fall back to `event_type from agent` when no reason was logged
        # (e.g., older records from before the selector existed).
        if report.root_cause_reason:
            parts.append(f"rooted in {report.root_cause_reason}")
        else:
            root = _resolve_event(xai, report.root_cause_event_id)
            if root:
                agent = root.get("agent_id") or "unknown"
                etype = root.get("event_type") or "event"
                parts.append(f"rooted in {etype} from {agent}")

    parts.append(f"{len(report.causal_chain)}-step causal chain")
    return "; ".join(parts) + "."
