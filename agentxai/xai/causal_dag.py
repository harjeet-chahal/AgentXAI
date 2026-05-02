"""
Pillar 6 — Temporal Causality.

CausalDAGBuilder reads every trajectory event, tool call, and message for a
task and constructs a NetworkX DiGraph whose nodes are TrajectoryEvent ids
and whose edges carry (weight, causal_type) attributes. Three edge sources:

    (a) Temporal precedence within an agent  (weight 0.3, "contributory")
    (b) Message causation: sender's most-recent event → receiver's next event
        (weight from the counterfactual_runs table if available, else from
        the message's acted_upon flag; type "direct")
    (c) Tool call → agent's next action
        (weight = ToolUseEvent.downstream_impact_score; type "direct")

When an edge is added twice, "direct" always upgrades over "contributory";
within the same type the higher weight wins.

Every edge in the final graph is persisted as a CausalEdge via
TrajectoryStore.save_causal_edge. The graph is validated as acyclic; any
cycle found is broken by removing the weakest edge (lowest weight).

`render_dot(graph) -> str` produces a simple Graphviz DOT serialization for
later visualization — no external Graphviz runtime required.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

import networkx as nx
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from agentxai.data.schemas import CausalEdge, ToolUseEvent, TrajectoryEvent
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.config import DEFAULT_CONFIG, XAIScoringConfig


_log = logging.getLogger(__name__)

# Backward-compat alias. Newly-written code should reach for
# `self.config.temporal_edge_weight` instead.
_TEMPORAL_WEIGHT: float = DEFAULT_CONFIG.temporal_edge_weight
_TOOL_EVENT_TYPES = {"tool_start", "tool_end", "tool_call", "tool_use"}


class CausalDAGBuilder:
    """Builds and persists the per-task causal DAG."""

    def __init__(
        self,
        store: TrajectoryStore,
        config: Optional[XAIScoringConfig] = None,
    ) -> None:
        self.store = store
        # Scoring weights — defaults reproduce historical behaviour when
        # `config` is None. See agentxai/xai/config.py.
        self.config: XAIScoringConfig = config or DEFAULT_CONFIG

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, task_id: str) -> nx.DiGraph:
        """Read the task's XAI data, build the DAG, persist its edges, return it."""
        record = self.store.get_full_record(task_id)
        events = sorted(record.xai_data.trajectory, key=lambda e: e.timestamp)

        graph: nx.DiGraph = nx.DiGraph()
        if not events:
            return graph

        for e in events:
            graph.add_node(
                e.event_id,
                agent_id=e.agent_id,
                event_type=e.event_type,
                action=e.action,
                timestamp=e.timestamp,
            )

        by_agent: Dict[str, List[TrajectoryEvent]] = defaultdict(list)
        for e in events:
            by_agent[e.agent_id].append(e)

        self._add_temporal_edges(graph, by_agent)
        self._add_message_edges(graph, record.xai_data.messages, by_agent, task_id)
        self._add_tool_edges(graph, record.xai_data.tool_calls, by_agent)

        self._break_cycles(graph)
        self._persist_edges(graph, task_id)
        return graph

    # ------------------------------------------------------------------
    # Edge construction
    # ------------------------------------------------------------------

    def _add_temporal_edges(
        self,
        graph: nx.DiGraph,
        by_agent: Dict[str, List[TrajectoryEvent]],
    ) -> None:
        for agent_events in by_agent.values():
            for a, b in zip(agent_events, agent_events[1:]):
                if a.event_id == b.event_id:
                    continue
                self._add_or_upgrade_edge(
                    graph, a.event_id, b.event_id,
                    weight=self.config.temporal_edge_weight,
                    causal_type="contributory",
                )

    def _add_message_edges(
        self,
        graph: nx.DiGraph,
        messages,
        by_agent: Dict[str, List[TrajectoryEvent]],
        task_id: str,
    ) -> None:
        cf_deltas = self._cf_deltas_for(task_id, "message")
        for m in messages:
            if not m.sender or not m.receiver:
                continue
            src = _latest_at_or_before(by_agent.get(m.sender, []), m.timestamp)
            tgt = _earliest_at_or_after(by_agent.get(m.receiver, []), m.timestamp)
            if src is None or tgt is None or src.event_id == tgt.event_id:
                continue
            weight = cf_deltas.get(m.message_id)
            if weight is None:
                # Heuristic when no counterfactual run is logged for
                # this message. The cf delta from the counterfactual_runs
                # table always wins when present.
                weight = (
                    self.config.message_acted_upon_weight
                    if m.acted_upon
                    else self.config.message_ignored_weight
                )
            self._add_or_upgrade_edge(
                graph, src.event_id, tgt.event_id,
                weight=float(weight), causal_type="direct",
            )

    def _add_tool_edges(
        self,
        graph: nx.DiGraph,
        tool_calls,
        by_agent: Dict[str, List[TrajectoryEvent]],
    ) -> None:
        for t in tool_calls:
            agent_events = by_agent.get(t.called_by, [])
            if not agent_events:
                continue
            tool_event = _match_tool_event(agent_events, t)
            if tool_event is None:
                continue
            next_action = _next_after(agent_events, tool_event)
            if next_action is None or next_action.event_id == tool_event.event_id:
                continue
            self._add_or_upgrade_edge(
                graph, tool_event.event_id, next_action.event_id,
                weight=float(t.downstream_impact_score),
                causal_type="direct",
            )

    def _add_or_upgrade_edge(
        self,
        graph: nx.DiGraph,
        u: str,
        v: str,
        *,
        weight: float,
        causal_type: str,
    ) -> None:
        """Insert an edge, or upgrade an existing one if the new edge is stronger."""
        if graph.has_edge(u, v):
            data = graph[u][v]
            existing_type = data.get("causal_type", "")
            existing_weight = data.get("weight", 0.0)
            direct_upgrade = causal_type == "direct" and existing_type != "direct"
            same_type_stronger = causal_type == existing_type and weight > existing_weight
            if direct_upgrade or same_type_stronger:
                data["weight"] = float(weight)
                data["causal_type"] = causal_type
        else:
            graph.add_edge(
                u, v,
                edge_id=str(uuid.uuid4()),
                weight=float(weight),
                causal_type=causal_type,
            )

    # ------------------------------------------------------------------
    # Cycle handling
    # ------------------------------------------------------------------

    def _break_cycles(self, graph: nx.DiGraph) -> None:
        """While a cycle exists, log a warning and remove the weakest edge in it."""
        while not nx.is_directed_acyclic_graph(graph):
            try:
                cycle = nx.find_cycle(graph)
            except nx.NetworkXNoCycle:  # pragma: no cover - defensive
                return
            edges = [(e[0], e[1]) for e in cycle]
            u, v = min(edges, key=lambda uv: graph[uv[0]][uv[1]].get("weight", 0.0))
            weight = graph[u][v].get("weight", 0.0)
            _log.warning(
                "Causal DAG cycle detected; removing weakest edge %s → %s (w=%.3f)",
                u, v, weight,
            )
            graph.remove_edge(u, v)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_edges(self, graph: nx.DiGraph, task_id: str) -> None:
        for u, v, data in graph.edges(data=True):
            edge = CausalEdge(
                edge_id=data.get("edge_id") or str(uuid.uuid4()),
                cause_event_id=u,
                effect_event_id=v,
                causal_strength=float(data.get("weight", 0.0)),
                causal_type=data.get("causal_type", ""),
            )
            self.store.save_causal_edge(task_id, edge)

    def _cf_deltas_for(self, task_id: str, perturbation_type: str) -> Dict[str, float]:
        """Load outcome_delta values from the counterfactual_runs table, if present."""
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


# ---------------------------------------------------------------------------
# Event-matching helpers
# ---------------------------------------------------------------------------

def _latest_at_or_before(
    events: List[TrajectoryEvent], ts: float,
) -> Optional[TrajectoryEvent]:
    candidates = [e for e in events if e.timestamp <= ts]
    return candidates[-1] if candidates else None


def _earliest_at_or_after(
    events: List[TrajectoryEvent], ts: float,
) -> Optional[TrajectoryEvent]:
    for e in events:
        if e.timestamp >= ts:
            return e
    return None


def _match_tool_event(
    agent_events: List[TrajectoryEvent], tool_call: ToolUseEvent,
) -> Optional[TrajectoryEvent]:
    """Pick the agent's event that most plausibly represents this tool invocation."""
    preferred = [e for e in agent_events if e.event_type in _TOOL_EVENT_TYPES
                 or (tool_call.tool_name and e.action == tool_call.tool_name)]
    candidates = preferred or list(agent_events)
    if not candidates:
        return None
    return min(candidates, key=lambda e: abs(e.timestamp - tool_call.timestamp))


def _next_after(
    agent_events: List[TrajectoryEvent], anchor: TrajectoryEvent,
) -> Optional[TrajectoryEvent]:
    for e in agent_events:
        if e.timestamp > anchor.timestamp:
            return e
    return None


# ---------------------------------------------------------------------------
# DOT rendering
# ---------------------------------------------------------------------------

def render_dot(graph: nx.DiGraph) -> str:
    """Serialize `graph` to a minimal Graphviz DOT string."""
    lines: List[str] = ["digraph causal_dag {", '  rankdir="LR";']
    for node, attrs in graph.nodes(data=True):
        label_parts = []
        if attrs.get("agent_id"):
            label_parts.append(str(attrs["agent_id"]))
        if attrs.get("action"):
            label_parts.append(str(attrs["action"]))
        elif attrs.get("event_type"):
            label_parts.append(str(attrs["event_type"]))
        label = " | ".join(label_parts) or str(node)
        lines.append(f'  "{node}" [label="{_escape(label)}"];')
    for u, v, attrs in graph.edges(data=True):
        weight = float(attrs.get("weight", 0.0))
        ctype = str(attrs.get("causal_type", ""))
        style = "solid" if ctype == "direct" else "dashed"
        lines.append(
            f'  "{u}" -> "{v}" '
            f'[label="{ctype} {weight:.2f}", weight={weight:.2f}, style={style}];'
        )
    lines.append("}")
    return "\n".join(lines)


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')
