"""
Tests for agentxai/xai/plan_tracker.py — Pillar 2.

The LLM is always mocked; no real network calls.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from agentxai.data.schemas import AgentXAIRecord, TrajectoryEvent
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.plan_tracker import (
    PlanTracker,
    _parse_reasons,
    _symmetric_diff,
)


TASK_ID = "PLAN-TEST-001"


class _FakeLLM:
    """Minimal chat-model stand-in that returns a canned AIMessage-like object."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: List[Any] = []

    def invoke(self, prompt: Any) -> Any:
        self.calls.append(prompt)
        return SimpleNamespace(content=self._content)


@pytest.fixture()
def store() -> TrajectoryStore:
    s = TrajectoryStore(db_url="sqlite:///:memory:")
    s.save_task(AgentXAIRecord(task_id=TASK_ID, source="test"))
    return s


@pytest.fixture()
def tracker(store: TrajectoryStore) -> PlanTracker:
    return PlanTracker(store=store, task_id=TASK_ID, llm=_FakeLLM('["r1", "r2"]'))


# ---------------------------------------------------------------------------
# register_plan
# ---------------------------------------------------------------------------

class TestRegisterPlan:
    def test_creates_and_persists(self, tracker, store):
        plan = tracker.register_plan("orchestrator", ["a", "b", "c"])

        assert plan.agent_id == "orchestrator"
        assert plan.intended_actions == ["a", "b", "c"]
        assert plan.actual_actions == []
        assert plan.deviations == []
        assert plan.deviation_reasons == []

        full = store.get_full_record(TASK_ID)
        assert len(full.xai_data.plans) == 1
        assert full.xai_data.plans[0].plan_id == plan.plan_id

    def test_intended_actions_is_copied(self, tracker):
        intended = ["a", "b"]
        plan = tracker.register_plan("x", intended)
        intended.append("c")
        assert plan.intended_actions == ["a", "b"]


# ---------------------------------------------------------------------------
# record_actual_action
# ---------------------------------------------------------------------------

class TestRecordActualAction:
    def test_appends_and_persists(self, tracker, store):
        plan = tracker.register_plan("agent_x", ["a", "b"])
        tracker.record_actual_action(plan.plan_id, "a")
        tracker.record_actual_action(plan.plan_id, "b")

        full = store.get_full_record(TASK_ID)
        assert full.xai_data.plans[0].actual_actions == ["a", "b"]

    def test_unknown_plan_id_raises(self, tracker):
        with pytest.raises(KeyError):
            tracker.record_actual_action("nonexistent", "whatever")


# ---------------------------------------------------------------------------
# finalize_plan
# ---------------------------------------------------------------------------

class TestFinalizePlan:
    def test_no_deviations_no_llm_call(self, store):
        llm = _FakeLLM('["should not be called"]')
        tracker = PlanTracker(store=store, task_id=TASK_ID, llm=llm)

        plan = tracker.register_plan("agent_x", ["a", "b"])
        tracker.record_actual_action(plan.plan_id, "a")
        tracker.record_actual_action(plan.plan_id, "b")

        finalized = tracker.finalize_plan(plan.plan_id)
        assert finalized.deviations == []
        assert finalized.deviation_reasons == []
        assert llm.calls == []  # LLM not invoked when there's nothing to explain

    def test_deviations_and_reasons_attached(self, store):
        llm = _FakeLLM('["specialist_b timed out", "synthesizer chose to retry"]')
        tracker = PlanTracker(store=store, task_id=TASK_ID, llm=llm)

        plan = tracker.register_plan(
            "orchestrator", ["route_a", "route_b", "synthesize"]
        )
        tracker.record_actual_action(plan.plan_id, "route_a")
        tracker.record_actual_action(plan.plan_id, "synthesize")
        tracker.record_actual_action(plan.plan_id, "retry")

        finalized = tracker.finalize_plan(plan.plan_id)

        # Missing ("route_b") first, then unexpected ("retry").
        assert finalized.deviations == ["route_b", "retry"]
        assert finalized.deviation_reasons == [
            "specialist_b timed out",
            "synthesizer chose to retry",
        ]

        # LLM was called exactly once with the prompt referencing both deviations.
        assert len(llm.calls) == 1
        prompt_text = str(llm.calls[0])
        assert "route_b" in prompt_text
        assert "retry" in prompt_text
        assert "orchestrator" in prompt_text

        # Persisted state matches the in-memory plan.
        full = store.get_full_record(TASK_ID)
        persisted = full.xai_data.plans[0]
        assert persisted.deviations == ["route_b", "retry"]
        assert persisted.deviation_reasons == [
            "specialist_b timed out",
            "synthesizer chose to retry",
        ]

    def test_trajectory_context_passed_to_llm(self, store):
        # Pre-populate a trajectory event for the agent so the prompt includes it.
        store.save_event(
            TASK_ID,
            TrajectoryEvent(
                agent_id="orchestrator",
                event_type="tool_end",
                action="route_b",
                outcome="TIMEOUT after 30s",
            ),
        )
        llm = _FakeLLM('["timed out"]')
        tracker = PlanTracker(store=store, task_id=TASK_ID, llm=llm)

        plan = tracker.register_plan("orchestrator", ["route_a", "route_b"])
        tracker.record_actual_action(plan.plan_id, "route_a")
        tracker.finalize_plan(plan.plan_id)

        prompt_text = str(llm.calls[0])
        assert "TIMEOUT" in prompt_text

    def test_llm_response_in_markdown(self, store):
        llm = _FakeLLM('Sure, here is my answer:\n```json\n["reason_one"]\n```')
        tracker = PlanTracker(store=store, task_id=TASK_ID, llm=llm)

        plan = tracker.register_plan("agent_x", ["a", "b"])
        tracker.record_actual_action(plan.plan_id, "a")

        finalized = tracker.finalize_plan(plan.plan_id)
        assert finalized.deviations == ["b"]
        assert finalized.deviation_reasons == ["reason_one"]

    def test_llm_error_falls_back_to_placeholder(self, store):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("api down")
        tracker = PlanTracker(store=store, task_id=TASK_ID, llm=llm)

        plan = tracker.register_plan("agent_x", ["a"])
        finalized = tracker.finalize_plan(plan.plan_id)
        assert finalized.deviations == ["a"]
        assert len(finalized.deviation_reasons) == 1
        assert "error" in finalized.deviation_reasons[0].lower()

    def test_llm_unavailable_returns_placeholders(self, store):
        tracker = PlanTracker(store=store, task_id=TASK_ID, llm=None)
        # Force the auto-init to have left llm unset.
        tracker.llm = None

        plan = tracker.register_plan("agent_x", ["a", "b"])
        finalized = tracker.finalize_plan(plan.plan_id)
        assert finalized.deviations == ["a", "b"]
        assert len(finalized.deviation_reasons) == 2
        for reason in finalized.deviation_reasons:
            assert "unavailable" in reason.lower()

    def test_unknown_plan_id_raises(self, tracker):
        with pytest.raises(KeyError):
            tracker.finalize_plan("nonexistent")


# ---------------------------------------------------------------------------
# _symmetric_diff
# ---------------------------------------------------------------------------

class TestSymmetricDiff:
    def test_only_missing(self):
        assert _symmetric_diff(["a", "b", "c"], ["a", "c"]) == ["b"]

    def test_only_extras(self):
        assert _symmetric_diff(["a"], ["a", "b"]) == ["b"]

    def test_missing_then_extras_order(self):
        assert _symmetric_diff(["a", "b"], ["a", "c"]) == ["b", "c"]

    def test_preserves_intended_order(self):
        assert _symmetric_diff(["x", "y", "z"], []) == ["x", "y", "z"]

    def test_empty(self):
        assert _symmetric_diff([], []) == []

    def test_identical_lists(self):
        assert _symmetric_diff(["a", "b"], ["a", "b"]) == []


# ---------------------------------------------------------------------------
# _parse_reasons
# ---------------------------------------------------------------------------

class TestParseReasons:
    def test_clean_json_array(self):
        assert _parse_reasons('["a", "b"]', 2) == ["a", "b"]

    def test_json_embedded_in_prose(self):
        # Valid JSON array (double-quoted) embedded in surrounding prose.
        out = _parse_reasons('intro text ["x", "y"] outro text', 2)
        assert out == ["x", "y"]

    def test_markdown_fenced(self):
        assert _parse_reasons('```json\n["x"]\n```', 1) == ["x"]

    def test_pads_short_response(self):
        out = _parse_reasons('["only_one"]', 3)
        assert len(out) == 3
        assert out[0] == "only_one"
        assert out[1] == "No explanation."

    def test_truncates_long_response(self):
        assert _parse_reasons('["a", "b", "c"]', 2) == ["a", "b"]

    def test_bullet_fallback(self):
        out = _parse_reasons("- first reason\n- second reason", 2)
        assert out == ["first reason", "second reason"]

    def test_empty_string(self):
        out = _parse_reasons("", 2)
        assert out == ["No explanation.", "No explanation."]
