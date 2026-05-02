"""Round-trip serialization tests for every schema in agentxai/data/schemas.py."""

import json
import pytest

from agentxai.data.schemas import (
    TrajectoryEvent,
    AgentPlan,
    ToolUseEvent,
    MemoryDiff,
    AgentMessage,
    CausalEdge,
    AccountabilityReport,
    CausalGraph,
    XAIData,
    AgentXAIRecord,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def roundtrip(cls, instance):
    """Serialise → JSON string → deserialise and return the recovered instance."""
    raw = instance.to_dict()
    # Ensure the dict is JSON-serialisable
    json_str = json.dumps(raw)
    recovered_dict = json.loads(json_str)
    return cls.from_dict(recovered_dict)


# ---------------------------------------------------------------------------
# Pillar 1 — TrajectoryEvent
# ---------------------------------------------------------------------------

class TestTrajectoryEvent:
    def _make(self, **kw) -> TrajectoryEvent:
        return TrajectoryEvent(
            agent_id="orchestrator",
            event_type="action",
            state_before={"step": 0},
            action="route_to_specialist_a",
            action_inputs={"case_id": "T001"},
            state_after={"step": 1},
            outcome="success",
            **kw,
        )

    def test_defaults_populated(self):
        e = self._make()
        assert e.event_id != ""
        assert e.timestamp > 0

    def test_roundtrip(self):
        original = self._make()
        recovered = roundtrip(TrajectoryEvent, original)
        assert recovered.event_id == original.event_id
        assert recovered.agent_id == original.agent_id
        assert recovered.event_type == original.event_type
        assert recovered.state_before == original.state_before
        assert recovered.action == original.action
        assert recovered.action_inputs == original.action_inputs
        assert recovered.state_after == original.state_after
        assert recovered.outcome == original.outcome
        assert recovered.timestamp == pytest.approx(original.timestamp)

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        expected = {
            "event_id", "timestamp", "agent_id", "event_type",
            "state_before", "action", "action_inputs", "state_after", "outcome",
        }
        assert set(d.keys()) == expected

    def test_unique_ids(self):
        assert self._make().event_id != self._make().event_id


# ---------------------------------------------------------------------------
# Pillar 2 — AgentPlan
# ---------------------------------------------------------------------------

class TestAgentPlan:
    def _make(self, **kw) -> AgentPlan:
        return AgentPlan(
            agent_id="specialist_a",
            intended_actions=["lookup_chest_pain", "score_severity", "store_findings"],
            actual_actions=["lookup_chest_pain", "store_findings"],
            deviations=["score_severity"],
            deviation_reasons=["severity_scorer returned error"],
            **kw,
        )

    def test_defaults_populated(self):
        p = self._make()
        assert p.plan_id != ""
        assert p.timestamp > 0

    def test_roundtrip(self):
        original = self._make()
        recovered = roundtrip(AgentPlan, original)
        assert recovered.plan_id == original.plan_id
        assert recovered.agent_id == original.agent_id
        assert recovered.intended_actions == original.intended_actions
        assert recovered.actual_actions == original.actual_actions
        assert recovered.deviations == original.deviations
        assert recovered.deviation_reasons == original.deviation_reasons

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        expected = {
            "plan_id", "agent_id", "timestamp", "intended_actions",
            "actual_actions", "deviations", "deviation_reasons",
        }
        assert set(d.keys()) == expected


# ---------------------------------------------------------------------------
# Pillar 3 — ToolUseEvent
# ---------------------------------------------------------------------------

class TestToolUseEvent:
    def _make(self, **kw) -> ToolUseEvent:
        return ToolUseEvent(
            tool_name="symptom_lookup",
            called_by="specialist_a",
            inputs={"symptom": "chest_pain"},
            outputs={"conditions": ["MI", "PE"], "scores": [0.82, 0.11]},
            duration_ms=142.7,
            downstream_impact_score=0.73,
            counterfactual_run_id="cf-run-001",
            **kw,
        )

    def test_defaults_populated(self):
        t = self._make()
        assert t.tool_call_id != ""
        assert t.timestamp > 0

    def test_roundtrip(self):
        original = self._make()
        recovered = roundtrip(ToolUseEvent, original)
        assert recovered.tool_call_id == original.tool_call_id
        assert recovered.tool_name == original.tool_name
        assert recovered.called_by == original.called_by
        assert recovered.inputs == original.inputs
        assert recovered.outputs == original.outputs
        assert recovered.duration_ms == pytest.approx(original.duration_ms)
        assert recovered.downstream_impact_score == pytest.approx(original.downstream_impact_score)
        assert recovered.counterfactual_run_id == original.counterfactual_run_id

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        expected = {
            "tool_call_id", "tool_name", "called_by", "timestamp",
            "inputs", "outputs", "duration_ms",
            "downstream_impact_score", "counterfactual_run_id",
        }
        assert set(d.keys()) == expected


# ---------------------------------------------------------------------------
# Pillar 4 — MemoryDiff
# ---------------------------------------------------------------------------

class TestMemoryDiff:
    def _make(self, **kw) -> MemoryDiff:
        return MemoryDiff(
            agent_id="specialist_a",
            operation="write",
            key="severity_score",
            value_before=None,
            value_after=0.82,
            triggered_by_event_id="evt-abc-123",
            **kw,
        )

    def test_defaults_populated(self):
        m = self._make()
        assert m.diff_id != ""
        assert m.timestamp > 0

    def test_roundtrip(self):
        original = self._make()
        recovered = roundtrip(MemoryDiff, original)
        assert recovered.diff_id == original.diff_id
        assert recovered.agent_id == original.agent_id
        assert recovered.operation == original.operation
        assert recovered.key == original.key
        assert recovered.value_before == original.value_before
        assert recovered.value_after == pytest.approx(original.value_after)
        assert recovered.triggered_by_event_id == original.triggered_by_event_id

    def test_roundtrip_complex_value(self):
        """value_before/value_after can hold nested dicts."""
        original = MemoryDiff(
            agent_id="specialist_b",
            operation="write",
            key="retrieved_docs",
            value_before=[],
            value_after=[{"doc_id": "d42", "score": 0.9}],
            triggered_by_event_id="evt-xyz",
        )
        recovered = roundtrip(MemoryDiff, original)
        assert recovered.value_after == original.value_after

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        expected = {
            "diff_id", "agent_id", "timestamp", "operation",
            "key", "value_before", "value_after", "triggered_by_event_id",
        }
        assert set(d.keys()) == expected


# ---------------------------------------------------------------------------
# Pillar 5 — AgentMessage
# ---------------------------------------------------------------------------

class TestAgentMessage:
    def _make(self, **kw) -> AgentMessage:
        return AgentMessage(
            sender="specialist_a",
            receiver="orchestrator",
            message_type="finding",
            content={
                "top_conditions": ["MI", "PE", "GERD"],
                "severity": 0.82,
                "confidence": 0.87,
            },
            acted_upon=True,
            behavior_change_description="Orchestrator updated routing priority to MI.",
            **kw,
        )

    def test_defaults_populated(self):
        msg = self._make()
        assert msg.message_id != ""
        assert msg.timestamp > 0

    def test_roundtrip(self):
        original = self._make()
        recovered = roundtrip(AgentMessage, original)
        assert recovered.message_id == original.message_id
        assert recovered.sender == original.sender
        assert recovered.receiver == original.receiver
        assert recovered.message_type == original.message_type
        assert recovered.content == original.content
        assert recovered.acted_upon == original.acted_upon
        assert recovered.behavior_change_description == original.behavior_change_description

    def test_roundtrip_not_acted_upon(self):
        original = AgentMessage(
            sender="specialist_b",
            receiver="orchestrator",
            message_type="finding",
            content={"guideline_match": "AHA_2023"},
            acted_upon=False,
            behavior_change_description="",
        )
        recovered = roundtrip(AgentMessage, original)
        assert recovered.acted_upon is False
        assert recovered.behavior_change_description == ""

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        expected = {
            "message_id", "sender", "receiver", "timestamp",
            "message_type", "content", "acted_upon", "behavior_change_description",
        }
        assert set(d.keys()) == expected


# ---------------------------------------------------------------------------
# Pillar 6 — CausalEdge
# ---------------------------------------------------------------------------

class TestCausalEdge:
    def _make(self, **kw) -> CausalEdge:
        return CausalEdge(
            cause_event_id="evt-001",
            effect_event_id="evt-007",
            causal_strength=0.91,
            causal_type="direct",
            **kw,
        )

    def test_defaults_populated(self):
        e = self._make()
        assert e.edge_id != ""

    def test_roundtrip(self):
        original = self._make()
        recovered = roundtrip(CausalEdge, original)
        assert recovered.edge_id == original.edge_id
        assert recovered.cause_event_id == original.cause_event_id
        assert recovered.effect_event_id == original.effect_event_id
        assert recovered.causal_strength == pytest.approx(original.causal_strength)
        assert recovered.causal_type == original.causal_type

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        expected = {
            "edge_id", "cause_event_id", "effect_event_id",
            "causal_strength", "causal_type",
        }
        assert set(d.keys()) == expected


# ---------------------------------------------------------------------------
# Pillar 7 — AccountabilityReport
# ---------------------------------------------------------------------------

class TestAccountabilityReport:
    def _make(self, **kw) -> AccountabilityReport:
        return AccountabilityReport(
            task_id="T001",
            final_outcome="MI",
            outcome_correct=True,
            agent_responsibility_scores={
                "specialist_a": 0.61,
                "specialist_b": 0.27,
                "synthesizer": 0.12,
            },
            root_cause_event_id="evt-001",
            causal_chain=["evt-001", "evt-003", "evt-007"],
            most_impactful_tool_call_id="tc-042",
            critical_memory_diffs=["diff-001", "diff-005"],
            most_influential_message_id="msg-003",
            plan_deviation_summary="specialist_a skipped score_severity due to tool error",
            one_line_explanation=(
                "Symptom pattern analysis (specialist_a) was the primary driver of the MI diagnosis."
            ),
            **kw,
        )

    def test_defaults_when_empty(self):
        r = AccountabilityReport()
        assert r.task_id != ""
        assert r.agent_responsibility_scores == {}
        assert r.causal_chain == []

    def test_roundtrip(self):
        original = self._make()
        recovered = roundtrip(AccountabilityReport, original)
        assert recovered.task_id == original.task_id
        assert recovered.final_outcome == original.final_outcome
        assert recovered.outcome_correct == original.outcome_correct
        assert recovered.agent_responsibility_scores == original.agent_responsibility_scores
        assert recovered.root_cause_event_id == original.root_cause_event_id
        assert recovered.causal_chain == original.causal_chain
        assert recovered.most_impactful_tool_call_id == original.most_impactful_tool_call_id
        assert recovered.critical_memory_diffs == original.critical_memory_diffs
        assert recovered.most_influential_message_id == original.most_influential_message_id
        assert recovered.plan_deviation_summary == original.plan_deviation_summary
        assert recovered.one_line_explanation == original.one_line_explanation

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        expected = {
            "task_id", "final_outcome", "outcome_correct",
            "agent_responsibility_scores", "root_cause_event_id",
            "root_cause_reason", "causal_chain",
            "most_impactful_tool_call_id", "critical_memory_diffs",
            "most_influential_message_id", "plan_deviation_summary",
            "one_line_explanation", "memory_usage",
            "evidence_used_by_final_answer", "most_supportive_evidence_ids",
        }
        assert set(d.keys()) == expected


# ---------------------------------------------------------------------------
# CausalGraph
# ---------------------------------------------------------------------------

class TestCausalGraph:
    def _make(self) -> CausalGraph:
        return CausalGraph(
            nodes=["evt-001", "evt-003", "evt-007"],
            edges=[
                CausalEdge(
                    cause_event_id="evt-001",
                    effect_event_id="evt-003",
                    causal_strength=0.8,
                    causal_type="direct",
                ),
                CausalEdge(
                    cause_event_id="evt-003",
                    effect_event_id="evt-007",
                    causal_strength=0.6,
                    causal_type="mediated",
                ),
            ],
        )

    def test_roundtrip(self):
        original = self._make()
        raw = original.to_dict()
        json_str = json.dumps(raw)
        recovered = CausalGraph.from_dict(json.loads(json_str))
        assert recovered.nodes == original.nodes
        assert len(recovered.edges) == len(original.edges)
        for r_edge, o_edge in zip(recovered.edges, original.edges):
            assert r_edge.edge_id == o_edge.edge_id
            assert r_edge.causal_strength == pytest.approx(o_edge.causal_strength)
            assert r_edge.causal_type == o_edge.causal_type

    def test_empty_graph(self):
        g = CausalGraph()
        raw = g.to_dict()
        recovered = CausalGraph.from_dict(raw)
        assert recovered.nodes == []
        assert recovered.edges == []


# ---------------------------------------------------------------------------
# XAIData
# ---------------------------------------------------------------------------

class TestXAIData:
    def _make(self) -> XAIData:
        te = TrajectoryEvent(agent_id="orchestrator", event_type="plan", action="emit_plan")
        ap = AgentPlan(agent_id="orchestrator", intended_actions=["route"])
        tc = ToolUseEvent(tool_name="symptom_lookup", called_by="specialist_a")
        md = MemoryDiff(agent_id="specialist_a", operation="write", key="severity_score",
                        value_after=0.82)
        msg = AgentMessage(sender="specialist_a", receiver="orchestrator",
                           message_type="finding", content={"severity": 0.82})
        cg = CausalGraph(
            nodes=[te.event_id],
            edges=[CausalEdge(cause_event_id=te.event_id, effect_event_id="evt-x",
                              causal_strength=0.5, causal_type="direct")],
        )
        ar = AccountabilityReport(
            task_id="T001",
            final_outcome="MI",
            outcome_correct=True,
            one_line_explanation="Driven by specialist_a severity score.",
        )
        return XAIData(
            trajectory=[te],
            plans=[ap],
            tool_calls=[tc],
            memory_diffs=[md],
            messages=[msg],
            causal_graph=cg,
            accountability_report=ar,
        )

    def test_roundtrip(self):
        original = self._make()
        raw = original.to_dict()
        json_str = json.dumps(raw)
        recovered = XAIData.from_dict(json.loads(json_str))

        assert len(recovered.trajectory) == 1
        assert recovered.trajectory[0].agent_id == "orchestrator"
        assert len(recovered.plans) == 1
        assert len(recovered.tool_calls) == 1
        assert len(recovered.memory_diffs) == 1
        assert len(recovered.messages) == 1
        assert len(recovered.causal_graph.nodes) == 1
        assert len(recovered.causal_graph.edges) == 1
        assert recovered.accountability_report is not None
        assert recovered.accountability_report.final_outcome == "MI"

    def test_none_accountability_report(self):
        xai = XAIData()
        raw = xai.to_dict()
        assert raw["accountability_report"] is None
        recovered = XAIData.from_dict(json.loads(json.dumps(raw)))
        assert recovered.accountability_report is None

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        expected = {
            "trajectory", "plans", "tool_calls", "memory_diffs",
            "messages", "causal_graph", "accountability_report",
        }
        assert set(d.keys()) == expected


# ---------------------------------------------------------------------------
# Top-level — AgentXAIRecord
# ---------------------------------------------------------------------------

class TestAgentXAIRecord:
    def _make(self) -> AgentXAIRecord:
        return AgentXAIRecord(
            task_id="T001",
            source="medqa",
            input={
                "patient_case": "A 45-year-old man presents with chest pain...",
                "answer_options": {"A": "MI", "B": "GERD", "C": "PE", "D": "Anxiety"},
            },
            ground_truth={
                "correct_answer": "A",
                "explanation": "ST elevation and troponin rise indicate MI.",
            },
            system_output={
                "final_diagnosis": "MI",
                "confidence": 0.88,
                "correct": True,
            },
            xai_data=XAIData(
                trajectory=[TrajectoryEvent(agent_id="orchestrator", event_type="plan",
                                            action="emit_plan")],
                accountability_report=AccountabilityReport(
                    task_id="T001",
                    final_outcome="MI",
                    outcome_correct=True,
                    one_line_explanation="Specialist A was decisive.",
                ),
            ),
        )

    def test_defaults_populated(self):
        r = AgentXAIRecord()
        assert r.task_id != ""
        assert r.source == "medqa"

    def test_roundtrip(self):
        original = self._make()
        raw = original.to_dict()
        json_str = json.dumps(raw)
        recovered = AgentXAIRecord.from_dict(json.loads(json_str))

        assert recovered.task_id == original.task_id
        assert recovered.source == original.source
        assert recovered.input == original.input
        assert recovered.ground_truth == original.ground_truth
        assert recovered.system_output == original.system_output
        assert len(recovered.xai_data.trajectory) == 1
        assert recovered.xai_data.accountability_report.final_outcome == "MI"

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        expected = {
            "task_id", "source", "input", "ground_truth", "system_output", "xai_data",
        }
        assert set(d.keys()) == expected

    def test_xai_data_keys(self):
        d = self._make().to_dict()
        expected = {
            "trajectory", "plans", "tool_calls", "memory_diffs",
            "messages", "causal_graph", "accountability_report",
        }
        assert set(d["xai_data"].keys()) == expected

    def test_json_serialisable(self):
        """to_dict() output must survive a full JSON round-trip without error."""
        original = self._make()
        json_str = json.dumps(original.to_dict())
        recovered = AgentXAIRecord.from_dict(json.loads(json_str))
        assert recovered.task_id == original.task_id
