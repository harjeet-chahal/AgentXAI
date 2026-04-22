"""
Tests for agentxai/xai/counterfactual_engine.py — Pillar 8.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from agentxai.data.schemas import AgentXAIRecord, ToolUseEvent
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.counterfactual_engine import (
    CounterfactualEngine,
    Pipeline,
    _neutral_baseline,
    _outcome_delta,
)


TASK_ID = "CF-TEST-001"


# ---------------------------------------------------------------------------
# Deterministic mock pipeline
# ---------------------------------------------------------------------------

class MockPipeline:
    """Returns a fixed `perturbed_output` regardless of snapshot/overrides."""

    def __init__(self, perturbed_output: Dict[str, Any]) -> None:
        self.perturbed_output = perturbed_output
        self.calls: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    def resume_from(
        self,
        state_snapshot: Dict[str, Any],
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.calls.append((state_snapshot, overrides))
        return dict(self.perturbed_output)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store() -> TrajectoryStore:
    s = TrajectoryStore(db_url="sqlite:///:memory:")
    s.save_task(AgentXAIRecord(task_id=TASK_ID, source="test"))
    return s


@pytest.fixture()
def tool_call(store: TrajectoryStore) -> ToolUseEvent:
    tc = ToolUseEvent(
        tool_name="symptom_lookup",
        called_by="specialist_a",
        inputs={"symptom": "chest_pain"},
        outputs={"conditions": ["MI", "PE"]},
        duration_ms=10.0,
    )
    store.save_tool_call(TASK_ID, tc)
    return tc


ORIGINAL = {"final_diagnosis": "MI", "confidence": 0.9}


# ---------------------------------------------------------------------------
# MockPipeline satisfies the Protocol
# ---------------------------------------------------------------------------

def test_mock_pipeline_satisfies_protocol():
    assert isinstance(MockPipeline({"final_diagnosis": "x", "confidence": 0.0}), Pipeline)


# ---------------------------------------------------------------------------
# Baseline / outcome helpers
# ---------------------------------------------------------------------------

class TestNeutralBaseline:
    def test_dict(self):
        assert _neutral_baseline({"a": 1}) == {}

    def test_list(self):
        assert _neutral_baseline([1, 2, 3]) == []

    def test_float(self):
        assert _neutral_baseline(3.14) == 0.0

    def test_int(self):
        assert _neutral_baseline(7) == 0.0

    def test_other(self):
        assert _neutral_baseline("hi") == {}


class TestOutcomeDelta:
    def test_identical(self):
        assert _outcome_delta(ORIGINAL, dict(ORIGINAL)) == 0.0

    def test_diagnosis_changed(self):
        d = _outcome_delta(ORIGINAL, {"final_diagnosis": "PE", "confidence": 0.9})
        assert d == 1.0

    def test_confidence_only_delta(self):
        d = _outcome_delta(ORIGINAL, {"final_diagnosis": "MI", "confidence": 0.4})
        assert d == pytest.approx(0.5)

    def test_capped_at_one(self):
        d = _outcome_delta(ORIGINAL, {"final_diagnosis": "PE", "confidence": 0.1})
        assert d == 1.0


# ---------------------------------------------------------------------------
# perturb_tool_output
# ---------------------------------------------------------------------------

class TestPerturbToolOutput:
    def test_same_outcome_scores_zero(self, store, tool_call):
        pipe = MockPipeline(dict(ORIGINAL))
        engine = CounterfactualEngine(
            store=store, pipeline=pipe, task_id=TASK_ID,
            state_snapshot={"ok": True}, original_output=ORIGINAL,
        )
        score = engine.perturb_tool_output(tool_call.tool_call_id)
        assert score == 0.0

    def test_changed_outcome_in_range(self, store, tool_call):
        pipe = MockPipeline({"final_diagnosis": "PE", "confidence": 0.7})
        engine = CounterfactualEngine(
            store=store, pipeline=pipe, task_id=TASK_ID,
            state_snapshot={}, original_output=ORIGINAL,
        )
        score = engine.perturb_tool_output(tool_call.tool_call_id)
        assert 0.0 <= score <= 1.0
        assert score == 1.0  # capped: diff dx (1.0) + |0.9-0.7|=0.2 → min(1, 1.2)

    def test_passes_baseline_via_overrides(self, store, tool_call):
        pipe = MockPipeline(dict(ORIGINAL))
        engine = CounterfactualEngine(
            store=store, pipeline=pipe, task_id=TASK_ID,
            state_snapshot={"s": 1}, original_output=ORIGINAL,
        )
        engine.perturb_tool_output(tool_call.tool_call_id)
        snap, overrides = pipe.calls[-1]
        assert snap == {"s": 1}
        assert overrides == {"tool_output": {tool_call.tool_call_id: {}}}

    def test_unknown_tool_id_raises(self, store):
        pipe = MockPipeline(dict(ORIGINAL))
        engine = CounterfactualEngine(
            store=store, pipeline=pipe, task_id=TASK_ID,
            original_output=ORIGINAL,
        )
        with pytest.raises(KeyError, match="not found"):
            engine.perturb_tool_output("does-not-exist")


# ---------------------------------------------------------------------------
# perturb_agent_output
# ---------------------------------------------------------------------------

class TestPerturbAgentOutput:
    def test_same_outcome_scores_zero(self, store):
        pipe = MockPipeline(dict(ORIGINAL))
        engine = CounterfactualEngine(
            store=store, pipeline=pipe, task_id=TASK_ID,
            original_output=ORIGINAL,
        )
        assert engine.perturb_agent_output("specialist_a") == 0.0

    def test_changed_outcome_in_range(self, store):
        pipe = MockPipeline({"final_diagnosis": "PE", "confidence": 0.5})
        engine = CounterfactualEngine(
            store=store, pipeline=pipe, task_id=TASK_ID,
            original_output=ORIGINAL,
        )
        score = engine.perturb_agent_output("specialist_a")
        assert 0.0 <= score <= 1.0

    def test_confidence_only_shift(self, store):
        pipe = MockPipeline({"final_diagnosis": "MI", "confidence": 0.6})
        engine = CounterfactualEngine(
            store=store, pipeline=pipe, task_id=TASK_ID,
            original_output=ORIGINAL,
        )
        score = engine.perturb_agent_output("specialist_a")
        assert score == pytest.approx(0.3)

    def test_overrides_carry_empty_memory(self, store):
        pipe = MockPipeline(dict(ORIGINAL))
        engine = CounterfactualEngine(
            store=store, pipeline=pipe, task_id=TASK_ID,
            original_output=ORIGINAL,
        )
        engine.perturb_agent_output("specialist_b")
        _, overrides = pipe.calls[-1]
        assert overrides == {"agent_memory": {"specialist_b": {}}}


# ---------------------------------------------------------------------------
# perturb_message
# ---------------------------------------------------------------------------

class TestPerturbMessage:
    def test_unchanged(self, store):
        pipe = MockPipeline(dict(ORIGINAL))
        engine = CounterfactualEngine(
            store=store, pipeline=pipe, task_id=TASK_ID,
            original_output=ORIGINAL,
        )
        changed, description = engine.perturb_message("msg-1")
        assert changed is False
        assert "no behavior change" in description

    def test_changed(self, store):
        pipe = MockPipeline({"final_diagnosis": "PE", "confidence": 0.6})
        engine = CounterfactualEngine(
            store=store, pipeline=pipe, task_id=TASK_ID,
            original_output=ORIGINAL,
        )
        changed, description = engine.perturb_message("msg-2")
        assert changed is True
        assert "MI" in description and "PE" in description

    def test_overrides_carry_neutral_payload(self, store):
        pipe = MockPipeline(dict(ORIGINAL))
        engine = CounterfactualEngine(
            store=store, pipeline=pipe, task_id=TASK_ID,
            original_output=ORIGINAL,
        )
        engine.perturb_message("msg-3")
        _, overrides = pipe.calls[-1]
        assert overrides == {"message_content": {"msg-3": {"info": "no information"}}}


# ---------------------------------------------------------------------------
# counterfactual_runs table
# ---------------------------------------------------------------------------

class TestCounterfactualRunsTable:
    def test_each_perturbation_logs_a_row(self, store, tool_call):
        pipe = MockPipeline({"final_diagnosis": "PE", "confidence": 0.7})
        engine = CounterfactualEngine(
            store=store, pipeline=pipe, task_id=TASK_ID,
            original_output=ORIGINAL,
        )
        engine.perturb_tool_output(tool_call.tool_call_id)
        engine.perturb_agent_output("specialist_a")
        engine.perturb_message("msg-1")

        runs = engine.list_runs()
        assert len(runs) == 3
        types = {r["perturbation_type"] for r in runs}
        assert types == {"tool_output", "agent_output", "message"}
        for r in runs:
            assert 0.0 <= r["outcome_delta"] <= 1.0
            assert r["task_id"] == TASK_ID
            assert r["original_outcome"] == ORIGINAL
            assert r["perturbed_outcome"] == {"final_diagnosis": "PE", "confidence": 0.7}
            assert r["run_id"]

    def test_scores_in_zero_one_range_across_many_runs(self, store, tool_call):
        """Sweep several deterministic fake outputs; every score must stay in [0, 1]."""
        fakes = [
            {"final_diagnosis": "MI", "confidence": 0.9},   # identical   → 0.0
            {"final_diagnosis": "MI", "confidence": 0.5},   # conf only   → 0.4
            {"final_diagnosis": "PE", "confidence": 0.9},   # dx only     → 1.0
            {"final_diagnosis": "PE", "confidence": 0.0},   # both, cap   → 1.0
        ]
        for fake in fakes:
            pipe = MockPipeline(fake)
            engine = CounterfactualEngine(
                store=store, pipeline=pipe, task_id=TASK_ID,
                original_output=ORIGINAL,
            )
            s1 = engine.perturb_tool_output(tool_call.tool_call_id)
            s2 = engine.perturb_agent_output("specialist_a")
            changed, _ = engine.perturb_message("msg-x")
            assert 0.0 <= s1 <= 1.0
            assert 0.0 <= s2 <= 1.0
            assert isinstance(changed, bool)
