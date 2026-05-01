"""
Pillar 7 — System-Wide Accountability.

AccountabilityReportGenerator reads every artefact produced by the other six
pillars and synthesises a single `AccountabilityReport`:

    * agent_responsibility_scores — one counterfactual re-run per specialist
      via `CounterfactualEngine.perturb_agent_output`, normalized to sum to 1.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence

import networkx as nx
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from agentxai.data.schemas import (
    AccountabilityReport,
    AgentPlan,
    AgentXAIRecord,
    CausalEdge,
    TrajectoryEvent,
    XAIData,
)
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.causal_dag import CausalDAGBuilder
from agentxai.xai.counterfactual_engine import CounterfactualEngine, Pipeline

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
    ) -> None:
        self.store = store
        self.pipeline = pipeline
        self._specialists = list(specialist_agents) if specialist_agents else None
        self.model = model

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

        graph = self._get_or_build_graph(record, task_id)
        terminal_id = self._find_terminal_event_id(xai)

        responsibility = self._responsibility_scores(
            task_id=task_id,
            xai=xai,
            state_snapshot=state_snapshot,
            original_output=original_output or dict(record.system_output),
        )

        root_cause_id = self._root_cause(graph, terminal_id)
        chain = self._causal_chain(graph, root_cause_id, terminal_id)

        tool_id = _most_impactful_tool(xai)
        critical_diffs = _critical_memory_diffs(xai, chain)
        cf_msg_deltas = self._cf_deltas_for(task_id, "message")
        msg_id = _most_influential_message(xai, cf_msg_deltas)
        deviation_summary = _deviation_summary(xai.plans)

        system_output = record.system_output or {}
        final_outcome = str(system_output.get("final_diagnosis", ""))
        outcome_correct = bool(system_output.get("correct", False))

        report = AccountabilityReport(
            task_id=task_id,
            final_outcome=final_outcome,
            outcome_correct=outcome_correct,
            agent_responsibility_scores=responsibility,
            root_cause_event_id=root_cause_id,
            causal_chain=chain,
            most_impactful_tool_call_id=tool_id,
            critical_memory_diffs=critical_diffs,
            most_influential_message_id=msg_id,
            plan_deviation_summary=deviation_summary,
            one_line_explanation="",
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
        return CausalDAGBuilder(self.store).build(task_id)

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

    def _root_cause(self, graph: nx.DiGraph, terminal_id: str) -> str:
        if not terminal_id or terminal_id not in graph:
            return ""
        reachers = nx.ancestors(graph, terminal_id)
        if not reachers:
            return ""

        def _out_weight_to_terminal(u: str) -> float:
            total = 0.0
            for v in graph.successors(u):
                if v == terminal_id or v in reachers:
                    total += float(graph[u][v].get("weight", 0.0))
            return total

        scored = [(u, _out_weight_to_terminal(u)) for u in reachers]
        best_score = max(s for _, s in scored)
        tied = [u for u, s in scored if s == best_score]
        # Tie-break: earliest timestamp (most upstream cause).
        return min(tied, key=lambda u: graph.nodes[u].get("timestamp", float("inf")))

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
    ) -> Dict[str, float]:
        specialists = self._resolve_specialists(xai)
        if not specialists:
            return {}
        if self.pipeline is None:
            # Can't run counterfactuals — distribute equally so the field is
            # still populated for downstream consumers.
            share = 1.0 / len(specialists)
            return {a: share for a in specialists}

        engine = CounterfactualEngine(
            store=self.store,
            pipeline=self.pipeline,
            task_id=task_id,
            state_snapshot=state_snapshot,
            original_output=original_output,
        )
        raw: Dict[str, float] = {}
        for agent in specialists:
            try:
                raw[agent] = float(engine.perturb_agent_output(agent))
            except Exception as exc:  # pragma: no cover - defensive
                _log.warning("Responsibility perturbation failed for %s: %s", agent, exc)
                raw[agent] = 0.0

        return _normalize_to_one(raw)

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
            return _fallback_explanation(report, xai)
        prompt = _build_explanation_prompt(report, xai)
        try:
            response = self.llm.invoke(prompt)
            text_out = _extract_text(response).strip()
        except Exception:
            return _fallback_explanation(report, xai)
        return text_out or _fallback_explanation(report, xai)


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

_TIE_EPSILON = 0.05


def _tied_top_agents(scores: Dict[str, float]) -> List[str]:
    """Return all agents within `_TIE_EPSILON` of the top score, ranked."""
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda kv: float(kv[1]), reverse=True)
    top = float(ranked[0][1])
    return [a for a, s in ranked if top - float(s) < _TIE_EPSILON]


def _build_explanation_prompt(report: AccountabilityReport, xai: XAIData) -> str:
    # Resolve raw UUIDs into descriptive dicts so the LLM never has to
    # reference an opaque id in its sentence.
    tool = _resolve_tool_call(xai, report.most_impactful_tool_call_id)
    # Suppress the tool field when its measured impact is effectively zero —
    # otherwise the LLM is invited to call a no-op tool "impactful".
    if tool and float(tool.get("impact_score") or 0.0) <= 0.0:
        tool = None

    scores = report.agent_responsibility_scores or {}
    tied = _tied_top_agents(scores)

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
    report: AccountabilityReport, xai: Optional[XAIData] = None
) -> str:
    correct = "correct" if report.outcome_correct else "incorrect"
    parts: List[str] = [f"Diagnosis {report.final_outcome!r} was {correct}"]

    scores = report.agent_responsibility_scores or {}
    tied = _tied_top_agents(scores)
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
        root = _resolve_event(xai, report.root_cause_event_id)
        if root:
            agent = root.get("agent_id") or "unknown"
            etype = root.get("event_type") or "event"
            parts.append(f"rooted in {etype} from {agent}")

    parts.append(f"{len(report.causal_chain)}-step causal chain")
    return "; ".join(parts) + "."
