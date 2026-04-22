"""
Tests for agentxai/xai/tool_provenance.py — Pillar 3.
"""

from __future__ import annotations

import time

import pytest

from agentxai.data.schemas import AgentXAIRecord
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.tool_provenance import (
    ToolProvenanceLogger,
    _capture_inputs,
    _capture_outputs,
    traced_tool,
)


TASK_ID = "TOOL-TEST-001"


@pytest.fixture()
def store() -> TrajectoryStore:
    s = TrajectoryStore(db_url="sqlite:///:memory:")
    s.save_task(AgentXAIRecord(task_id=TASK_ID, source="test"))
    return s


@pytest.fixture()
def logger(store: TrajectoryStore) -> ToolProvenanceLogger:
    return ToolProvenanceLogger(store=store, task_id=TASK_ID)


# ---------------------------------------------------------------------------
# log_tool_call
# ---------------------------------------------------------------------------

class TestLogToolCall:
    def test_writes_event_and_persists(self, logger, store):
        event = logger.log_tool_call(
            tool_name="symptom_lookup",
            called_by="specialist_a",
            inputs={"symptom": "chest_pain"},
            outputs={"conditions": ["MI", "PE"]},
            duration_ms=123.4,
        )
        assert event.tool_call_id
        assert event.tool_name == "symptom_lookup"
        assert event.called_by == "specialist_a"
        assert event.inputs == {"symptom": "chest_pain"}
        assert event.outputs == {"conditions": ["MI", "PE"]}
        assert event.duration_ms == pytest.approx(123.4)
        # Not-yet-computed sentinel values
        assert event.downstream_impact_score == 0.0
        assert event.counterfactual_run_id == ""

        full = store.get_full_record(TASK_ID)
        assert len(full.xai_data.tool_calls) == 1
        persisted = full.xai_data.tool_calls[0]
        assert persisted.tool_call_id == event.tool_call_id
        assert persisted.inputs == event.inputs
        assert persisted.outputs == event.outputs

    def test_non_jsonable_values_are_stringified(self, logger, store):
        class Opaque:
            def __str__(self) -> str:
                return "<opaque>"

        logger.log_tool_call(
            tool_name="t",
            called_by="a",
            inputs={"blob": Opaque()},
            outputs={"result": Opaque()},
            duration_ms=1.0,
        )
        full = store.get_full_record(TASK_ID)
        tc = full.xai_data.tool_calls[0]
        assert tc.inputs == {"blob": "<opaque>"}
        assert tc.outputs == {"result": "<opaque>"}


# ---------------------------------------------------------------------------
# attach_impact_score
# ---------------------------------------------------------------------------

class TestAttachImpactScore:
    def test_updates_existing_event(self, logger, store):
        event = logger.log_tool_call(
            tool_name="t", called_by="a", inputs={}, outputs={}, duration_ms=1.0
        )
        updated = logger.attach_impact_score(
            event.tool_call_id, score=0.82, counterfactual_run_id="cf-001"
        )
        assert updated.downstream_impact_score == pytest.approx(0.82)
        assert updated.counterfactual_run_id == "cf-001"

        persisted = store.get_full_record(TASK_ID).xai_data.tool_calls[0]
        assert persisted.downstream_impact_score == pytest.approx(0.82)
        assert persisted.counterfactual_run_id == "cf-001"

    def test_works_on_event_not_in_memory_cache(self, store):
        """A fresh logger instance should still patch a previously-stored event."""
        first = ToolProvenanceLogger(store=store, task_id=TASK_ID)
        event = first.log_tool_call(
            tool_name="t", called_by="a", inputs={}, outputs={}, duration_ms=1.0
        )

        second = ToolProvenanceLogger(store=store, task_id=TASK_ID)
        second.attach_impact_score(event.tool_call_id, score=0.5, counterfactual_run_id="cf-x")

        persisted = store.get_full_record(TASK_ID).xai_data.tool_calls[0]
        assert persisted.downstream_impact_score == pytest.approx(0.5)
        assert persisted.counterfactual_run_id == "cf-x"

    def test_unknown_id_raises(self, logger):
        with pytest.raises(KeyError, match="not found"):
            logger.attach_impact_score("nonexistent", 0.1, "cf")


# ---------------------------------------------------------------------------
# @traced_tool decorator
# ---------------------------------------------------------------------------

class TestTracedTool:
    def test_wraps_and_logs(self, logger, store):
        @traced_tool(logger, called_by="specialist_a")
        def lookup_symptoms(query: str) -> dict:
            return {"conditions": ["MI", "PE"], "query": query}

        result = lookup_symptoms("chest pain")

        # Return value is passed through untouched.
        assert result == {"conditions": ["MI", "PE"], "query": "chest pain"}

        full = store.get_full_record(TASK_ID)
        assert len(full.xai_data.tool_calls) == 1
        tc = full.xai_data.tool_calls[0]
        assert tc.tool_name == "lookup_symptoms"
        assert tc.called_by == "specialist_a"
        assert tc.inputs == {"input": "chest pain"}
        assert tc.outputs == {"conditions": ["MI", "PE"], "query": "chest pain"}
        assert tc.duration_ms >= 0.0

    def test_explicit_tool_name(self, logger, store):
        @traced_tool(logger, called_by="a", tool_name="overridden")
        def some_fn(x):
            return x * 2

        some_fn(21)
        tc = store.get_full_record(TASK_ID).xai_data.tool_calls[0]
        assert tc.tool_name == "overridden"

    def test_dict_input_is_flattened(self, logger, store):
        @traced_tool(logger, called_by="a")
        def multi_arg_tool(payload):
            return "ok"

        multi_arg_tool({"symptom": "pain", "severity": 3})
        tc = store.get_full_record(TASK_ID).xai_data.tool_calls[0]
        assert tc.inputs == {"symptom": "pain", "severity": 3}

    def test_mixed_args_kwargs(self, logger, store):
        @traced_tool(logger, called_by="a")
        def fn(x, y, *, z):
            return x + y + z

        fn(1, 2, z=3)
        tc = store.get_full_record(TASK_ID).xai_data.tool_calls[0]
        assert tc.inputs == {"args": [1, 2], "kwargs": {"z": 3}}
        assert tc.outputs == {"output": 6}

    def test_duration_is_measured(self, logger, store):
        @traced_tool(logger, called_by="a")
        def slow_tool(x):
            time.sleep(0.02)  # 20 ms
            return x

        slow_tool("hi")
        tc = store.get_full_record(TASK_ID).xai_data.tool_calls[0]
        assert tc.duration_ms >= 15.0  # allow small slack under load

    def test_exception_is_logged_and_reraised(self, logger, store):
        @traced_tool(logger, called_by="a")
        def broken(x):
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            broken("nope")

        tc = store.get_full_record(TASK_ID).xai_data.tool_calls[0]
        assert tc.tool_name == "broken"
        assert tc.inputs == {"input": "nope"}
        assert "ValueError" in tc.outputs["error"]
        assert "boom" in tc.outputs["error"]

    def test_works_with_langchain_style_tool_func(self, logger, store):
        """
        Simulate LangChain's Tool(func=...) pattern: assign a wrapped function
        to the `.func` attribute of a Tool-like object and invoke it.
        """
        class _FakeTool:
            name: str = "lookup"

            def __init__(self, func):
                self.func = func

        def raw_lookup(query: str) -> str:
            return f"found: {query}"

        fake = _FakeTool(func=traced_tool(logger, called_by="specialist_a")(raw_lookup))
        out = fake.func("fever")
        assert out == "found: fever"

        tc = store.get_full_record(TASK_ID).xai_data.tool_calls[0]
        assert tc.tool_name == "raw_lookup"
        assert tc.called_by == "specialist_a"
        assert tc.inputs == {"input": "fever"}
        assert tc.outputs == {"output": "found: fever"}


# ---------------------------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------------------------

class TestCaptureHelpers:
    def test_single_scalar_arg(self):
        assert _capture_inputs(("hi",), {}) == {"input": "hi"}

    def test_single_dict_arg_flattens(self):
        assert _capture_inputs(({"a": 1},), {}) == {"a": 1}

    def test_kwargs_only(self):
        assert _capture_inputs((), {"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_mixed_goes_into_args_kwargs(self):
        out = _capture_inputs((1, 2), {"z": 3})
        assert out == {"args": [1, 2], "kwargs": {"z": 3}}

    def test_success_output(self):
        assert _capture_outputs(42, None) == {"output": 42}

    def test_dict_output(self):
        assert _capture_outputs({"k": 1}, None) == {"k": 1}

    def test_error_output(self):
        err = RuntimeError("oops")
        out = _capture_outputs(None, err)
        assert "RuntimeError" in out["error"]
        assert "oops" in out["error"]
