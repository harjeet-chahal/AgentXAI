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

    def test_non_jsonable_tool_output_is_structured(self, logger: TrajectoryLogger):
        # The OLD `_to_jsonable` collapsed unknown objects into `str(obj)`,
        # which silently stored useless `<MyClass object at 0x…>` strings.
        # The new fallback preserves the type name for forensic value.
        class Blob:
            def __str__(self) -> str:
                return "<blob>"

        logger.on_tool_end(Blob(), tags=["a"])
        out = logger.current_trajectory()[0].state_after["output"]
        assert isinstance(out, dict)
        assert out["__type__"].endswith(".Blob")
        # The repr is included so debuggers can still see object content.
        assert "Blob" in out["repr"]


# ---------------------------------------------------------------------------
# _to_jsonable — new structured fallback
# ---------------------------------------------------------------------------

class TestToJsonableFallback:
    """Coverage of the four ladder rungs in `_to_jsonable`."""

    def test_passthrough_for_jsonable_primitives(self):
        from agentxai.xai.trajectory_logger import _to_jsonable
        assert _to_jsonable("hi") == "hi"
        assert _to_jsonable(42) == 42
        assert _to_jsonable(3.14) == 3.14
        assert _to_jsonable(True) is True
        assert _to_jsonable(None) is None

    def test_recurses_into_dicts_and_lists(self):
        from agentxai.xai.trajectory_logger import _to_jsonable
        out = _to_jsonable({"k": [1, 2, {"nested": True}]})
        assert out == {"k": [1, 2, {"nested": True}]}

    def test_dict_keys_coerced_to_str(self):
        # JSON keys must be strings — non-string keys (e.g. ints from
        # tool outputs) get stringified rather than dropped.
        from agentxai.xai.trajectory_logger import _to_jsonable
        assert _to_jsonable({1: "a", 2: "b"}) == {"1": "a", "2": "b"}

    def test_sets_become_lists(self):
        from agentxai.xai.trajectory_logger import _to_jsonable
        out = _to_jsonable({1, 2, 3})
        assert isinstance(out, list)
        assert sorted(out) == [1, 2, 3]

    def test_dataclass_uses_asdict(self):
        # AgentXAI's own schemas are dataclasses — they must round-trip
        # cleanly through the asdict path.
        from dataclasses import dataclass

        from agentxai.xai.trajectory_logger import _to_jsonable

        @dataclass
        class Point:
            x: int
            y: int

        out = _to_jsonable(Point(1, 2))
        assert out == {"x": 1, "y": 2}

    def test_dataclass_class_itself_is_not_treated_as_instance(self):
        # `dataclasses.is_dataclass` is True for both classes AND
        # instances. We must only call asdict on instances; classes
        # should fall through to the structured fallback.
        from dataclasses import dataclass

        from agentxai.xai.trajectory_logger import _to_jsonable

        @dataclass
        class Empty:
            pass

        out = _to_jsonable(Empty)   # the class object, not an instance
        assert isinstance(out, dict)
        assert "__type__" in out and "repr" in out

    def test_pydantic_v2_model_dump(self):
        # Pydantic models (LangChain BaseMessage, FastAPI response
        # models, etc.) all expose `.model_dump()` in v2.
        from agentxai.xai.trajectory_logger import _to_jsonable

        class FakeV2Model:
            def model_dump(self):
                return {"role": "user", "content": "hi"}

        assert _to_jsonable(FakeV2Model()) == {"role": "user", "content": "hi"}

    def test_pydantic_v1_dict_method(self):
        from agentxai.xai.trajectory_logger import _to_jsonable

        class FakeV1Model:
            def dict(self):   # noqa: A003 — emulates pydantic v1 API
                return {"role": "system", "content": "x"}

        assert _to_jsonable(FakeV1Model()) == {"role": "system", "content": "x"}

    def test_to_dict_convention(self):
        from agentxai.xai.trajectory_logger import _to_jsonable

        class HasToDict:
            def to_dict(self):
                return {"kind": "custom", "value": 7}

        assert _to_jsonable(HasToDict()) == {"kind": "custom", "value": 7}

    def test_structured_fallback_for_unknown_object(self):
        # No dataclass, no pydantic, no convention helper → structured
        # fallback with the type name + repr.
        from agentxai.xai.trajectory_logger import _to_jsonable

        class Mystery:
            def __repr__(self):
                return "<Mystery 42>"

        out = _to_jsonable(Mystery())
        assert isinstance(out, dict)
        assert set(out.keys()) == {"__type__", "repr"}
        assert out["__type__"].endswith(".Mystery")
        assert out["repr"] == "<Mystery 42>"

    def test_structured_fallback_truncates_giant_repr(self):
        # A giant repr would blow up the JSON column. Cap it.
        from agentxai.xai.trajectory_logger import _to_jsonable

        class Giant:
            def __repr__(self):
                return "x" * 2000

        out = _to_jsonable(Giant())
        assert len(out["repr"]) <= 240
        assert out["repr"].endswith("...")

    def test_failed_dump_falls_through_to_structured(self):
        # A model whose model_dump() raises must NOT be lost — fall
        # through to the structured fallback.
        from agentxai.xai.trajectory_logger import _to_jsonable

        class BrokenModel:
            def model_dump(self):
                raise RuntimeError("boom")

        out = _to_jsonable(BrokenModel())
        assert isinstance(out, dict)
        assert "__type__" in out
