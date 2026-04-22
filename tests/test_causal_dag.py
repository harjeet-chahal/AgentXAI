"""
Tests for agentxai/xai/causal_dag.py — Pillar 6.
"""

from __future__ import annotations

import logging
import uuid
from typing import List

import networkx as nx
import pytest

from agentxai.data.schemas import (
    AgentMessage,
    AgentXAIRecord,
    ToolUseEvent,
    TrajectoryEvent,
)
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.causal_dag import CausalDAGBuilder, render_dot


TASK_ID = "DAG-TEST-001"


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------

def _mk_event(
    *, agent_id: str, event_type: str, timestamp: float, action: str = "",
) -> TrajectoryEvent:
    return TrajectoryEvent(
        event_id=str(uuid.uuid4()),
        timestamp=timestamp,
        agent_id=agent_id,
        event_type=event_type,
        action=action,
    )


@pytest.fixture()
def store() -> TrajectoryStore:
    s = TrajectoryStore(db_url="sqlite:///:memory:")
    s.save_task(AgentXAIRecord(task_id=TASK_ID, source="test"))
    return s


@pytest.fixture()
def scenario(store: TrajectoryStore):
    """
    Fabricate a 6-event trajectory with one message and one tool call.

        t=1.0  e1  orchestrator   plan            "triage"
        t=2.0  e2  orchestrator   agent_action    "route"
        t=3.0  e3  specialist_a   tool_start      "symptom_lookup"
        t=4.0  e4  specialist_a   tool_end        "symptom_lookup"
        t=5.0  e5  specialist_a   agent_action    "diagnose"
        t=6.0  e6  specialist_a   agent_action    "finalize"

        msg:  orchestrator → specialist_a   t=2.5   acted_upon=True
        tool: symptom_lookup (by specialist_a) t=3.1  impact=0.8
    """
    e1 = _mk_event(agent_id="orchestrator", event_type="plan",         timestamp=1.0, action="triage")
    e2 = _mk_event(agent_id="orchestrator", event_type="agent_action", timestamp=2.0, action="route")
    e3 = _mk_event(agent_id="specialist_a", event_type="tool_start",   timestamp=3.0, action="symptom_lookup")
    e4 = _mk_event(agent_id="specialist_a", event_type="tool_end",     timestamp=4.0, action="symptom_lookup")
    e5 = _mk_event(agent_id="specialist_a", event_type="agent_action", timestamp=5.0, action="diagnose")
    e6 = _mk_event(agent_id="specialist_a", event_type="agent_action", timestamp=6.0, action="finalize")
    events = [e1, e2, e3, e4, e5, e6]
    for e in events:
        store.save_event(TASK_ID, e)

    msg = AgentMessage(
        sender="orchestrator",
        receiver="specialist_a",
        timestamp=2.5,
        message_type="routing",
        content={"to": "specialist_a"},
        acted_upon=True,
    )
    store.save_message(TASK_ID, msg)

    tool = ToolUseEvent(
        tool_name="symptom_lookup",
        called_by="specialist_a",
        timestamp=3.1,
        inputs={"symptom": "chest_pain"},
        outputs={"conditions": ["MI"]},
        duration_ms=50.0,
        downstream_impact_score=0.8,
    )
    store.save_tool_call(TASK_ID, tool)

    return {"events": events, "msg": msg, "tool": tool}


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------

class TestBuild:
    def test_fabricated_6_event_trajectory(self, store, scenario):
        e1, e2, e3, e4, e5, e6 = scenario["events"]

        g = CausalDAGBuilder(store).build(TASK_ID)

        # Nodes: one per trajectory event.
        assert set(g.nodes) == {e.event_id for e in scenario["events"]}

        # (a) Temporal precedence within agent.
        # orchestrator: e1 → e2
        assert g.has_edge(e1.event_id, e2.event_id)
        assert g[e1.event_id][e2.event_id]["causal_type"] == "contributory"
        assert g[e1.event_id][e2.event_id]["weight"] == pytest.approx(0.3)

        # specialist_a temporal chain: e3→e4 (upgraded by tool), e4→e5, e5→e6.
        assert g.has_edge(e4.event_id, e5.event_id)
        assert g[e4.event_id][e5.event_id]["causal_type"] == "contributory"
        assert g[e4.event_id][e5.event_id]["weight"] == pytest.approx(0.3)

        assert g.has_edge(e5.event_id, e6.event_id)
        assert g[e5.event_id][e6.event_id]["causal_type"] == "contributory"
        assert g[e5.event_id][e6.event_id]["weight"] == pytest.approx(0.3)

        # (b) Message: orchestrator's latest ≤ 2.5 = e2; specialist_a's
        # earliest ≥ 2.5 = e3. acted_upon=True → weight 1.0.
        assert g.has_edge(e2.event_id, e3.event_id)
        assert g[e2.event_id][e3.event_id]["causal_type"] == "direct"
        assert g[e2.event_id][e3.event_id]["weight"] == pytest.approx(1.0)

        # (c) Tool → next action. Tool t=3.1 is closest to e3 (tool_start);
        # next event after e3 is e4. Contributory 0.3 upgrades to direct 0.8.
        assert g.has_edge(e3.event_id, e4.event_id)
        assert g[e3.event_id][e4.event_id]["causal_type"] == "direct"
        assert g[e3.event_id][e4.event_id]["weight"] == pytest.approx(0.8)

        # Exactly the 5 expected edges, nothing extra.
        assert g.number_of_edges() == 5

        # Acyclic.
        assert nx.is_directed_acyclic_graph(g)

    def test_edges_persisted_to_store(self, store, scenario):
        CausalDAGBuilder(store).build(TASK_ID)
        persisted = store.get_full_record(TASK_ID).xai_data.causal_graph
        assert len(persisted.edges) == 5
        endpoints = {(e.cause_event_id, e.effect_event_id) for e in persisted.edges}
        e1, e2, e3, e4, e5, e6 = scenario["events"]
        assert (e1.event_id, e2.event_id) in endpoints
        assert (e2.event_id, e3.event_id) in endpoints
        assert (e3.event_id, e4.event_id) in endpoints
        assert (e4.event_id, e5.event_id) in endpoints
        assert (e5.event_id, e6.event_id) in endpoints
        # Types are preserved.
        by_pair = {(e.cause_event_id, e.effect_event_id): e for e in persisted.edges}
        assert by_pair[(e1.event_id, e2.event_id)].causal_type == "contributory"
        assert by_pair[(e2.event_id, e3.event_id)].causal_type == "direct"
        assert by_pair[(e3.event_id, e4.event_id)].causal_type == "direct"
        assert by_pair[(e3.event_id, e4.event_id)].causal_strength == pytest.approx(0.8)

    def test_counterfactual_run_weight_takes_precedence(self, store, scenario):
        """If a counterfactual_runs row exists for a message, its outcome_delta
        is used as the edge weight instead of the acted_upon fallback."""
        from agentxai.xai.counterfactual_engine import CounterfactualEngine

        class _NoopPipeline:
            def resume_from(self, state_snapshot, overrides):
                return {"final_diagnosis": "MI", "confidence": 0.9}

        engine = CounterfactualEngine(
            store=store, pipeline=_NoopPipeline(), task_id=TASK_ID,
            original_output={"final_diagnosis": "MI", "confidence": 0.9},
        )
        # Force a known outcome_delta in the counterfactual_runs table.
        msg_id = scenario["msg"].message_id
        from sqlalchemy import text
        with store._engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO counterfactual_runs "
                    "(run_id, task_id, perturbation_type, target_id, "
                    " baseline_value_json, original_outcome_json, "
                    " perturbed_outcome_json, outcome_delta) "
                    "VALUES (:r, :t, 'message', :m, '{}', '{}', '{}', 0.42)"
                ),
                {"r": "cf-manual", "t": TASK_ID, "m": msg_id},
            )
            conn.commit()
        del engine  # table creation side effect is what we needed

        e1, e2, e3, e4, e5, e6 = scenario["events"]
        g = CausalDAGBuilder(store).build(TASK_ID)
        assert g[e2.event_id][e3.event_id]["weight"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Empty / edge-case inputs
# ---------------------------------------------------------------------------

class TestEmpty:
    def test_no_events_returns_empty_graph(self, store):
        g = CausalDAGBuilder(store).build(TASK_ID)
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0


# ---------------------------------------------------------------------------
# Cycle handling
# ---------------------------------------------------------------------------

class TestCycleBreaking:
    def test_weakest_edge_is_removed_on_cycle(self, store, caplog):
        builder = CausalDAGBuilder(store)
        g: nx.DiGraph = nx.DiGraph()
        g.add_node("a"); g.add_node("b"); g.add_node("c")
        g.add_edge("a", "b", edge_id="e1", weight=0.9, causal_type="direct")
        g.add_edge("b", "c", edge_id="e2", weight=0.8, causal_type="direct")
        g.add_edge("c", "a", edge_id="e3", weight=0.1, causal_type="contributory")

        with caplog.at_level(logging.WARNING, logger="agentxai.xai.causal_dag"):
            builder._break_cycles(g)

        assert nx.is_directed_acyclic_graph(g)
        assert not g.has_edge("c", "a")
        assert g.has_edge("a", "b") and g.has_edge("b", "c")
        assert any("cycle" in rec.message.lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# render_dot()
# ---------------------------------------------------------------------------

class TestRenderDot:
    def test_dot_contains_nodes_and_edges(self, store, scenario):
        g = CausalDAGBuilder(store).build(TASK_ID)
        dot = render_dot(g)
        assert dot.startswith("digraph causal_dag {")
        assert dot.rstrip().endswith("}")
        e1, e2, *_ = scenario["events"]
        assert f'"{e1.event_id}"' in dot
        assert f'"{e1.event_id}" -> "{e2.event_id}"' in dot
        assert "contributory" in dot
        assert "direct" in dot

    def test_dot_escapes_quotes_in_labels(self):
        g: nx.DiGraph = nx.DiGraph()
        g.add_node("n1", agent_id='ag"ent', action='do "x"')
        dot = render_dot(g)
        assert r'\"' in dot
