"""
Tests for agentxai/store/trajectory_store.py.

All tests use an in-memory SQLite database (no file I/O).
Each test class gets a fresh store via the `store` fixture.
"""

from __future__ import annotations

import pytest

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store() -> TrajectoryStore:
    """Fresh in-memory SQLite store for each test."""
    return TrajectoryStore(db_url="sqlite:///:memory:")


# Canonical fake task used across all tests
TASK_ID = "TEST-TASK-001"


def _make_record() -> AgentXAIRecord:
    """Build a fully populated AgentXAIRecord with one of every artefact type."""
    event = TrajectoryEvent(
        agent_id    = "orchestrator",
        event_type  = "plan",
        state_before= {"step": 0},
        action      = "emit_plan",
        action_inputs={"case_id": TASK_ID},
        state_after = {"step": 1},
        outcome     = "success",
    )
    plan = AgentPlan(
        agent_id         = "orchestrator",
        intended_actions = ["route_to_a", "route_to_b", "synthesize"],
        actual_actions   = ["route_to_a", "synthesize"],
        deviations       = ["route_to_b"],
        deviation_reasons= ["specialist_b timed out"],
    )
    tool_call = ToolUseEvent(
        tool_name               = "symptom_lookup",
        called_by               = "specialist_a",
        inputs                  = {"symptom": "chest_pain"},
        outputs                 = {"conditions": ["MI", "PE"]},
        duration_ms             = 123.4,
        downstream_impact_score = 0.82,
        counterfactual_run_id   = "cf-001",
    )
    diff = MemoryDiff(
        agent_id              = "specialist_a",
        operation             = "write",
        key                   = "severity_score",
        value_before          = None,
        value_after           = 0.87,
        triggered_by_event_id = event.event_id,
    )
    message = AgentMessage(
        sender                      = "specialist_a",
        receiver                    = "orchestrator",
        message_type                = "finding",
        content                     = {"top_conditions": ["MI"], "severity": 0.87},
        acted_upon                  = True,
        behavior_change_description = "Orchestrator updated routing priority.",
    )
    edge = CausalEdge(
        cause_event_id  = event.event_id,
        effect_event_id = "evt-downstream",
        causal_strength = 0.91,
        causal_type     = "direct",
    )
    report = AccountabilityReport(
        task_id                     = TASK_ID,
        final_outcome               = "MI",
        outcome_correct             = True,
        agent_responsibility_scores = {"specialist_a": 0.7, "synthesizer": 0.3},
        root_cause_event_id         = event.event_id,
        causal_chain                = [event.event_id, "evt-downstream"],
        most_impactful_tool_call_id = tool_call.tool_call_id,
        critical_memory_diffs       = [diff.diff_id],
        most_influential_message_id = message.message_id,
        plan_deviation_summary      = "specialist_b was skipped",
        one_line_explanation        = "Symptom analysis drove the MI diagnosis.",
    )
    return AgentXAIRecord(
        task_id      = TASK_ID,
        source       = "medqa",
        input        = {"patient_case": "45yo chest pain", "answer_options": {"A": "MI"}},
        ground_truth = {"correct_answer": "A", "explanation": "ST elevation"},
        system_output= {"final_diagnosis": "MI", "confidence": 0.88, "correct": True},
        xai_data     = XAIData(
            trajectory   = [event],
            plans        = [plan],
            tool_calls   = [tool_call],
            memory_diffs = [diff],
            messages     = [message],
            causal_graph = __import__("agentxai.data.schemas", fromlist=["CausalGraph"]).CausalGraph(
                nodes=[event.event_id, "evt-downstream"],
                edges=[edge],
            ),
            accountability_report=report,
        ),
    )


def _save_all(store: TrajectoryStore, record: AgentXAIRecord) -> None:
    """Persist every artefact in the record via individual save_* calls."""
    store.save_task(record)
    for ev in record.xai_data.trajectory:
        store.save_event(record.task_id, ev)
    for plan in record.xai_data.plans:
        store.save_plan(record.task_id, plan)
    for tc in record.xai_data.tool_calls:
        store.save_tool_call(record.task_id, tc)
    for diff in record.xai_data.memory_diffs:
        store.save_memory_diff(record.task_id, diff)
    for msg in record.xai_data.messages:
        store.save_message(record.task_id, msg)
    for edge in record.xai_data.causal_graph.edges:
        store.save_causal_edge(record.task_id, edge)
    if record.xai_data.accountability_report:
        store.save_accountability_report(record.xai_data.accountability_report)


# ---------------------------------------------------------------------------
# Schema / table creation
# ---------------------------------------------------------------------------

class TestStoreInit:
    def test_creates_without_error(self, store):
        assert store is not None

    def test_list_tasks_empty(self, store):
        assert store.list_tasks() == []

    def test_get_missing_task_raises(self, store):
        with pytest.raises(KeyError, match="not found"):
            store.get_full_record("DOES_NOT_EXIST")


# ---------------------------------------------------------------------------
# Individual writers
# ---------------------------------------------------------------------------

class TestSaveTask:
    def test_save_and_list(self, store):
        record = _make_record()
        store.save_task(record)
        tasks = store.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == TASK_ID
        assert tasks[0]["source"] == "medqa"

    def test_upsert_is_idempotent(self, store):
        record = _make_record()
        store.save_task(record)
        store.save_task(record)          # second call must not raise
        assert len(store.list_tasks()) == 1


class TestSaveEvent:
    def test_save_and_retrieve(self, store):
        record = _make_record()
        store.save_task(record)
        event = record.xai_data.trajectory[0]
        store.save_event(TASK_ID, event)

        full = store.get_full_record(TASK_ID)
        assert len(full.xai_data.trajectory) == 1
        recovered = full.xai_data.trajectory[0]
        assert recovered.event_id   == event.event_id
        assert recovered.agent_id   == event.agent_id
        assert recovered.event_type == event.event_type
        assert recovered.action     == event.action
        assert recovered.state_before == event.state_before
        assert recovered.state_after  == event.state_after
        assert recovered.outcome    == event.outcome

    def test_upsert_updates_outcome(self, store):
        record = _make_record()
        store.save_task(record)
        event = record.xai_data.trajectory[0]
        store.save_event(TASK_ID, event)
        event.outcome = "updated"
        store.save_event(TASK_ID, event)

        full = store.get_full_record(TASK_ID)
        assert full.xai_data.trajectory[0].outcome == "updated"


class TestSavePlan:
    def test_save_and_retrieve(self, store):
        record = _make_record()
        store.save_task(record)
        plan = record.xai_data.plans[0]
        store.save_plan(TASK_ID, plan)

        full = store.get_full_record(TASK_ID)
        assert len(full.xai_data.plans) == 1
        recovered = full.xai_data.plans[0]
        assert recovered.plan_id          == plan.plan_id
        assert recovered.agent_id         == plan.agent_id
        assert recovered.intended_actions == plan.intended_actions
        assert recovered.actual_actions   == plan.actual_actions
        assert recovered.deviations       == plan.deviations
        assert recovered.deviation_reasons== plan.deviation_reasons


class TestSaveToolCall:
    def test_save_and_retrieve(self, store):
        record = _make_record()
        store.save_task(record)
        tc = record.xai_data.tool_calls[0]
        store.save_tool_call(TASK_ID, tc)

        full = store.get_full_record(TASK_ID)
        assert len(full.xai_data.tool_calls) == 1
        recovered = full.xai_data.tool_calls[0]
        assert recovered.tool_call_id            == tc.tool_call_id
        assert recovered.tool_name               == tc.tool_name
        assert recovered.called_by               == tc.called_by
        assert recovered.inputs                  == tc.inputs
        assert recovered.outputs                 == tc.outputs
        assert recovered.duration_ms             == pytest.approx(tc.duration_ms)
        assert recovered.downstream_impact_score == pytest.approx(tc.downstream_impact_score)
        assert recovered.counterfactual_run_id   == tc.counterfactual_run_id


class TestSaveMemoryDiff:
    def test_save_and_retrieve(self, store):
        record = _make_record()
        store.save_task(record)
        diff = record.xai_data.memory_diffs[0]
        store.save_memory_diff(TASK_ID, diff)

        full = store.get_full_record(TASK_ID)
        assert len(full.xai_data.memory_diffs) == 1
        recovered = full.xai_data.memory_diffs[0]
        assert recovered.diff_id               == diff.diff_id
        assert recovered.agent_id              == diff.agent_id
        assert recovered.operation             == diff.operation
        assert recovered.key                   == diff.key
        assert recovered.value_before          == diff.value_before
        assert recovered.value_after           == pytest.approx(diff.value_after)
        assert recovered.triggered_by_event_id == diff.triggered_by_event_id

    def test_null_value_before_round_trips(self, store):
        record = _make_record()
        store.save_task(record)
        diff = record.xai_data.memory_diffs[0]
        diff.value_before = None
        store.save_memory_diff(TASK_ID, diff)

        full = store.get_full_record(TASK_ID)
        assert full.xai_data.memory_diffs[0].value_before is None

    def test_complex_value_after(self, store):
        record = _make_record()
        store.save_task(record)
        diff = record.xai_data.memory_diffs[0]
        diff.value_after = {"nested": [1, 2, 3], "flag": True}
        store.save_memory_diff(TASK_ID, diff)

        full = store.get_full_record(TASK_ID)
        assert full.xai_data.memory_diffs[0].value_after == diff.value_after


class TestSaveMessage:
    def test_save_and_retrieve(self, store):
        record = _make_record()
        store.save_task(record)
        msg = record.xai_data.messages[0]
        store.save_message(TASK_ID, msg)

        full = store.get_full_record(TASK_ID)
        assert len(full.xai_data.messages) == 1
        recovered = full.xai_data.messages[0]
        assert recovered.message_id                  == msg.message_id
        assert recovered.sender                      == msg.sender
        assert recovered.receiver                    == msg.receiver
        assert recovered.message_type                == msg.message_type
        assert recovered.content                     == msg.content
        assert recovered.acted_upon                  == msg.acted_upon
        assert recovered.behavior_change_description == msg.behavior_change_description

    def test_acted_upon_false(self, store):
        record = _make_record()
        store.save_task(record)
        msg = record.xai_data.messages[0]
        msg.acted_upon = False
        store.save_message(TASK_ID, msg)

        full = store.get_full_record(TASK_ID)
        assert full.xai_data.messages[0].acted_upon is False


class TestSaveCausalEdge:
    def test_save_and_retrieve(self, store):
        record = _make_record()
        store.save_task(record)
        edge = record.xai_data.causal_graph.edges[0]
        store.save_causal_edge(TASK_ID, edge)

        full = store.get_full_record(TASK_ID)
        assert len(full.xai_data.causal_graph.edges) == 1
        recovered = full.xai_data.causal_graph.edges[0]
        assert recovered.edge_id         == edge.edge_id
        assert recovered.cause_event_id  == edge.cause_event_id
        assert recovered.effect_event_id == edge.effect_event_id
        assert recovered.causal_strength == pytest.approx(edge.causal_strength)
        assert recovered.causal_type     == edge.causal_type


class TestSaveAccountabilityReport:
    def test_save_and_retrieve(self, store):
        record = _make_record()
        store.save_task(record)
        report = record.xai_data.accountability_report
        store.save_accountability_report(report)

        full = store.get_full_record(TASK_ID)
        r = full.xai_data.accountability_report
        assert r is not None
        assert r.task_id                     == report.task_id
        assert r.final_outcome               == report.final_outcome
        assert r.outcome_correct             == report.outcome_correct
        assert r.agent_responsibility_scores == report.agent_responsibility_scores
        assert r.root_cause_event_id         == report.root_cause_event_id
        assert r.causal_chain                == report.causal_chain
        assert r.most_impactful_tool_call_id == report.most_impactful_tool_call_id
        assert r.critical_memory_diffs       == report.critical_memory_diffs
        assert r.most_influential_message_id == report.most_influential_message_id
        assert r.plan_deviation_summary      == report.plan_deviation_summary
        assert r.one_line_explanation        == report.one_line_explanation

    def test_no_report_returns_none(self, store):
        record = _make_record()
        store.save_task(record)
        full = store.get_full_record(TASK_ID)
        assert full.xai_data.accountability_report is None


# ---------------------------------------------------------------------------
# Full round-trip: insert every artefact type, read back as AgentXAIRecord
# ---------------------------------------------------------------------------

class TestFullRoundTrip:
    def test_get_full_record_all_fields(self, store):
        record = _make_record()
        _save_all(store, record)

        recovered = store.get_full_record(TASK_ID)

        # Task metadata
        assert recovered.task_id      == record.task_id
        assert recovered.source       == record.source
        assert recovered.input        == record.input
        assert recovered.ground_truth == record.ground_truth
        assert recovered.system_output== record.system_output

        xai = recovered.xai_data

        # Pillar 1
        assert len(xai.trajectory) == 1
        assert xai.trajectory[0].agent_id == "orchestrator"

        # Pillar 2
        assert len(xai.plans) == 1
        assert xai.plans[0].deviations == ["route_to_b"]

        # Pillar 3
        assert len(xai.tool_calls) == 1
        assert xai.tool_calls[0].tool_name == "symptom_lookup"
        assert xai.tool_calls[0].downstream_impact_score == pytest.approx(0.82)

        # Pillar 4
        assert len(xai.memory_diffs) == 1
        assert xai.memory_diffs[0].value_after == pytest.approx(0.87)

        # Pillar 5
        assert len(xai.messages) == 1
        assert xai.messages[0].acted_upon is True

        # Pillar 6
        assert len(xai.causal_graph.edges) == 1
        assert xai.causal_graph.edges[0].causal_strength == pytest.approx(0.91)

        # Pillar 7
        assert xai.accountability_report is not None
        assert xai.accountability_report.final_outcome == "MI"
        assert xai.accountability_report.outcome_correct is True
        assert xai.accountability_report.agent_responsibility_scores == {
            "specialist_a": 0.7, "synthesizer": 0.3
        }

    def test_list_tasks_after_save(self, store):
        record = _make_record()
        _save_all(store, record)

        tasks = store.list_tasks()
        assert len(tasks) == 1
        t = tasks[0]
        assert t["task_id"]        == TASK_ID
        assert t["source"]         == "medqa"
        assert t["final_outcome"]  == "MI"
        assert t["outcome_correct"] is True

    def test_multiple_tasks(self, store):
        for i in range(3):
            rec = AgentXAIRecord(
                task_id="TASK-%03d" % i,
                source="medqa",
                input={"q": f"question_{i}"},
            )
            store.save_task(rec)

        tasks = store.list_tasks(limit=10)
        assert len(tasks) == 3

    def test_list_tasks_limit(self, store):
        for i in range(5):
            store.save_task(AgentXAIRecord(task_id=f"T-{i}", source="medqa"))
        assert len(store.list_tasks(limit=3)) == 3

    def test_trajectory_ordered_by_timestamp(self, store):
        """Events must come back in ascending timestamp order."""
        import time
        record = _make_record()
        store.save_task(record)

        e1 = TrajectoryEvent(agent_id="a", event_type="action", timestamp=1000.0)
        e2 = TrajectoryEvent(agent_id="b", event_type="action", timestamp=2000.0)
        e3 = TrajectoryEvent(agent_id="c", event_type="action", timestamp=1500.0)

        for ev in (e1, e2, e3):
            store.save_event(TASK_ID, ev)

        full = store.get_full_record(TASK_ID)
        timestamps = [ev.timestamp for ev in full.xai_data.trajectory]
        assert timestamps == sorted(timestamps)
