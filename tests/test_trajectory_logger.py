"""
Tests for agentxai/xai/trajectory_logger.py — Pillar 1.

We drive the callback surface directly (no LangChain required at test time)
and assert that events land in the store and come back in order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytest

from agentxai.data.schemas import AgentXAIRecord
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.trajectory_logger import TrajectoryLogger


TASK_ID = "TRAJ-TEST-001"


@dataclass
class _FakeAgentAction:
    """Stand-in for langchain_core.agents.AgentAction."""
    tool: str
    tool_input: Dict[str, Any]
    log: str = ""


@pytest.fixture()
def store() -> TrajectoryStore:
    return TrajectoryStore(db_url="sqlite:///:memory:")


@pytest.fixture()
def logger(store: TrajectoryStore) -> TrajectoryLogger:
    store.save_task(AgentXAIRecord(task_id=TASK_ID, source="test"))
    return TrajectoryLogger(store=store, task_id=TASK_ID)


# ---------------------------------------------------------------------------
# Direct log_event API
# ---------------------------------------------------------------------------

class TestLogEvent:
    def test_writes_and_returns_event(self, logger: TrajectoryLogger):
        ev = logger.log_event(
            agent_id="orchestrator",
            event_type="action",
            state_before={"step": 0},
            action="route",
            action_inputs={"to": "specialist_a"},
            state_after={"step": 1},
            outcome="success",
        )
        assert ev.agent_id == "orchestrator"
        assert ev.event_type == "action"
        assert ev.action == "route"
        assert ev.outcome == "success"
        assert ev.event_id
        assert ev.timestamp > 0

    def test_event_is_persisted(self, logger: TrajectoryLogger, store: TrajectoryStore):
        logger.log_event(agent_id="a", event_type="action")
        full = store.get_full_record(TASK_ID)
        assert len(full.xai_data.trajectory) == 1
        assert full.xai_data.trajectory[0].agent_id == "a"

    def test_current_trajectory_returns_all(self, logger: TrajectoryLogger):
        for i in range(3):
            logger.log_event(agent_id=f"a{i}", event_type="action")
        traj = logger.current_trajectory()
        assert len(traj) == 3
        assert [e.agent_id for e in traj] == ["a0", "a1", "a2"]

    def test_current_trajectory_ordered_by_timestamp(self, logger: TrajectoryLogger):
        e1 = logger.log_event(agent_id="a", event_type="action")
        e2 = logger.log_event(agent_id="b", event_type="action")
        e3 = logger.log_event(agent_id="c", event_type="action")

        e2.timestamp = 100.0
        e1.timestamp = 200.0
        e3.timestamp = 300.0

        traj = logger.current_trajectory()
        assert [e.agent_id for e in traj] == ["b", "a", "c"]


# ---------------------------------------------------------------------------
# LangChain callback surface
# ---------------------------------------------------------------------------

class TestLangChainCallbacks:
    def test_short_sequence_in_order(self, logger: TrajectoryLogger, store: TrajectoryStore):
        # Simulate: chain_start → agent_action → tool_start → tool_end → chain_end
        logger.on_chain_start(
            serialized={"name": "orchestrator_chain"},
            inputs={"case": "45yo chest pain"},
            tags=["orchestrator"],
        )
        logger.on_agent_action(
            _FakeAgentAction(
                tool="symptom_lookup",
                tool_input={"symptom": "chest_pain"},
                log="Decided to look up symptoms.",
            ),
            tags=["orchestrator"],
        )
        logger.on_tool_start(
            serialized={"name": "symptom_lookup"},
            input_str='{"symptom": "chest_pain"}',
            tags=["specialist_a"],
        )
        logger.on_tool_end(
            "{'conditions': ['MI', 'PE']}",
            tags=["specialist_a"],
        )
        logger.on_chain_end(
            {"final_diagnosis": "MI"},
            tags=["orchestrator"],
        )

        traj = logger.current_trajectory()
        assert [e.event_type for e in traj] == [
            "chain_start",
            "agent_action",
            "tool_start",
            "tool_end",
            "chain_end",
        ]

        # agent_ids were pulled from tags
        assert traj[0].agent_id == "orchestrator"
        assert traj[1].agent_id == "orchestrator"
        assert traj[2].agent_id == "specialist_a"
        assert traj[3].agent_id == "specialist_a"
        assert traj[4].agent_id == "orchestrator"

        # action / action_inputs round-tripped correctly
        assert traj[0].action == "orchestrator_chain"
        assert traj[1].action == "symptom_lookup"
        assert traj[1].action_inputs == {"symptom": "chest_pain"}
        assert traj[2].action == "symptom_lookup"
        assert traj[3].state_after == {"output": "{'conditions': ['MI', 'PE']}"}
        assert traj[4].state_after == {"final_diagnosis": "MI"}

        # Everything made it to the store in the same order
        persisted = store.get_full_record(TASK_ID).xai_data.trajectory
        assert [e.event_type for e in persisted] == [
            "chain_start",
            "agent_action",
            "tool_start",
            "tool_end",
            "chain_end",
        ]

    def test_agent_action_with_scalar_input(self, logger: TrajectoryLogger):
        logger.on_agent_action(
            _FakeAgentAction(tool="echo", tool_input="hello", log=""),
            tags=["agent_x"],
        )
        traj = logger.current_trajectory()
        assert traj[0].action == "echo"
        assert traj[0].action_inputs == {"input": "hello"}

    def test_agent_id_falls_back_to_metadata(self, logger: TrajectoryLogger):
        logger.on_tool_start(
            serialized={"name": "t"},
            input_str="x",
            metadata={"agent_id": "specialist_b"},
        )
        assert logger.current_trajectory()[0].agent_id == "specialist_b"

    def test_non_jsonable_tool_output_is_stringified(self, logger: TrajectoryLogger):
        class Blob:
            def __str__(self) -> str:
                return "<blob>"

        logger.on_tool_end(Blob(), tags=["a"])
        assert logger.current_trajectory()[0].state_after == {"output": "<blob>"}
