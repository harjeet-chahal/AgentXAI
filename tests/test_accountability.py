"""
Tests for agentxai/xai/accountability.py — Pillar 7.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Tuple

import pytest

from agentxai.data.schemas import (
    AgentMessage,
    AgentPlan,
    AgentXAIRecord,
    MemoryDiff,
    ToolUseEvent,
    TrajectoryEvent,
)
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.accountability import (
    AccountabilityReportGenerator,
    _deviation_summary,
    _fallback_explanation,
    _normalize_to_one,
)


TASK_ID = "ACCT-TEST-001"


# ---------------------------------------------------------------------------
# Fake LLM + pipeline
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = text


class FakeLLM:
    def __init__(self, text: str = "The synthesizer reached MI via specialist_a's findings.") -> None:
        self.text = text
        self.prompts: List[str] = []

    def invoke(self, prompt: str) -> _FakeResponse:
        self.prompts.append(prompt)
        return _FakeResponse(self.text)


class MockPipeline:
    """Returns a fixed, distinct-from-original output so every perturbation scores 1.0."""

    def __init__(self, output: Dict[str, Any]) -> None:
        self.output = output
        self.calls: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    def resume_from(self, state_snapshot, overrides):
        self.calls.append((state_snapshot, overrides))
        return dict(self.output)


# ---------------------------------------------------------------------------
# Fabricated XAI data
# ---------------------------------------------------------------------------

def _mk_event(agent_id, event_type, timestamp, action=""):
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
    s.save_task(AgentXAIRecord(
        task_id=TASK_ID,
        source="test",
        input={"patient_case": "chest pain, SOB"},
        ground_truth={"correct_answer": "B"},
        system_output={"final_diagnosis": "MI", "confidence": 0.9, "correct": True},
    ))
    return s


@pytest.fixture()
def scenario(store: TrajectoryStore):
    """
    t=1.0  e1  orchestrator   plan
    t=2.0  e2  orchestrator   agent_action    "route"          (sends m1)
    t=3.0  e3  specialist_a   tool_start      "symptom_lookup"
    t=4.0  e4  specialist_a   tool_end        "symptom_lookup"
    t=5.0  e5  specialist_a   agent_action    "diagnose"       (writes d1, sends m2)
    t=6.0  e6  specialist_b   agent_action    "consult"        (sends m3)
    t=7.0  e7  synthesizer    final_diagnosis "MI"
    """
    events = [
        _mk_event("orchestrator", "plan",            1.0, "triage"),
        _mk_event("orchestrator", "agent_action",    2.0, "route"),
        _mk_event("specialist_a", "tool_start",      3.0, "symptom_lookup"),
        _mk_event("specialist_a", "tool_end",        4.0, "symptom_lookup"),
        _mk_event("specialist_a", "agent_action",    5.0, "diagnose"),
        _mk_event("specialist_b", "agent_action",    6.0, "consult"),
        _mk_event("synthesizer",  "final_diagnosis", 7.0, "MI"),
    ]
    for e in events:
        store.save_event(TASK_ID, e)
    e1, e2, e3, e4, e5, e6, e7 = events

    # Messages
    m1 = AgentMessage(sender="orchestrator", receiver="specialist_a",
                      timestamp=2.5, message_type="routing",
                      content={"case_id": "p1"}, acted_upon=True)
    m2 = AgentMessage(sender="specialist_a", receiver="synthesizer",
                      timestamp=5.5, message_type="finding",
                      content={"dx": "MI"}, acted_upon=True)
    m3 = AgentMessage(sender="specialist_b", receiver="synthesizer",
                      timestamp=6.5, message_type="finding",
                      content={"dx": "PE"}, acted_upon=False)
    for m in (m1, m2, m3):
        store.save_message(TASK_ID, m)

    # Tool call — closest to e3 by timestamp.
    tool = ToolUseEvent(
        tool_name="symptom_lookup", called_by="specialist_a", timestamp=3.2,
        inputs={"symptom": "chest_pain"}, outputs={"conditions": ["MI"]},
        duration_ms=42.0, downstream_impact_score=0.9,
    )
    store.save_tool_call(TASK_ID, tool)

    # A low-impact tool call to verify we pick the highest.
    tool_low = ToolUseEvent(
        tool_name="severity_scorer", called_by="specialist_a", timestamp=4.2,
        inputs={"dx": "MI"}, outputs={"score": 0.2},
        duration_ms=5.0, downstream_impact_score=0.1,
    )
    store.save_tool_call(TASK_ID, tool_low)

    # Memory diff triggered by e5 (on the causal chain).
    d1 = MemoryDiff(
        agent_id="specialist_a", operation="write", key="diagnosis",
        value_before=None, value_after="MI", triggered_by_event_id=e5.event_id,
    )
    # A diff unrelated to the chain (triggered by a non-chain event).
    d2 = MemoryDiff(
        agent_id="specialist_b", operation="write", key="note",
        value_before=None, value_after="hmm", triggered_by_event_id="detached-event",
    )
    store.save_memory_diff(TASK_ID, d1)
    store.save_memory_diff(TASK_ID, d2)

    # Plan with a deviation for specialist_a.
    p = AgentPlan(
        agent_id="specialist_a",
        intended_actions=["symptom_lookup", "score_severity"],
        actual_actions=["symptom_lookup"],
        deviations=["score_severity"],
        deviation_reasons=["Skipped due to insufficient data."],
    )
    store.save_plan(TASK_ID, p)

    return {
        "events": events,
        "messages": {"m1": m1, "m2": m2, "m3": m3},
        "tool": tool,
        "tool_low": tool_low,
        "diffs": {"d1": d1, "d2": d2},
        "plan": p,
    }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_normalize_nonzero(self):
        assert _normalize_to_one({"a": 1.0, "b": 3.0}) == {"a": 0.25, "b": 0.75}

    def test_normalize_all_zero_distributes_equally(self):
        out = _normalize_to_one({"a": 0.0, "b": 0.0})
        assert out == {"a": 0.5, "b": 0.5}

    def test_normalize_empty(self):
        assert _normalize_to_one({}) == {}

    def test_deviation_summary_joins_per_agent(self):
        p1 = AgentPlan(agent_id="a", deviations=["X"], deviation_reasons=["because"])
        p2 = AgentPlan(agent_id="b", deviations=["Y"], deviation_reasons=["reasons"])
        out = _deviation_summary([p1, p2])
        assert out == "a: X — because\nb: Y — reasons"

    def test_fallback_explanation_mentions_top_agent(self):
        from agentxai.data.schemas import AccountabilityReport
        r = AccountabilityReport(
            final_outcome="MI", outcome_correct=True,
            agent_responsibility_scores={"specialist_a": 0.7, "specialist_b": 0.3},
            root_cause_event_id="e2", causal_chain=["e2", "e3", "e7"],
        )
        s = _fallback_explanation(r)
        assert "specialist_a" in s
        assert "MI" in s
        assert "correct" in s


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_all_fields_populated(self, store, scenario):
        pipeline = MockPipeline({"final_diagnosis": "PE", "confidence": 0.5})
        llm = FakeLLM("Synthesizer chose MI, anchored by specialist_a's tool output.")
        gen = AccountabilityReportGenerator(store=store, pipeline=pipeline, llm=llm)

        report = gen.generate(TASK_ID)

        assert report.task_id == TASK_ID
        assert report.final_outcome == "MI"
        assert report.outcome_correct is True

        # Two specialists → equal 0.5 shares (both perturbations change outcome).
        assert set(report.agent_responsibility_scores) == {"specialist_a", "specialist_b"}
        assert report.agent_responsibility_scores["specialist_a"] == pytest.approx(0.5)
        assert report.agent_responsibility_scores["specialist_b"] == pytest.approx(0.5)
        assert sum(report.agent_responsibility_scores.values()) == pytest.approx(1.0)

        # Root cause + chain: must be one of the ancestors; chain must end at e7.
        e1, e2, e3, e4, e5, e6, e7 = scenario["events"]
        ancestors = {e1.event_id, e2.event_id, e3.event_id, e4.event_id, e5.event_id, e6.event_id}
        assert report.root_cause_event_id in ancestors
        assert len(report.causal_chain) >= 2
        assert report.causal_chain[0] == report.root_cause_event_id
        assert report.causal_chain[-1] == e7.event_id

        # Highest-impact tool = the one with score 0.9.
        assert report.most_impactful_tool_call_id == scenario["tool"].tool_call_id

        # Only d1 (triggered by e5) should be critical if e5 is on the chain.
        if e5.event_id in report.causal_chain:
            assert report.critical_memory_diffs == [scenario["diffs"]["d1"].diff_id]
        else:
            assert scenario["diffs"]["d2"].diff_id not in report.critical_memory_diffs

        # Acted-upon messages (weight 1.0) beat the non-acted message (0.5).
        assert report.most_influential_message_id in {
            scenario["messages"]["m1"].message_id,
            scenario["messages"]["m2"].message_id,
        }

        # Deviation summary populated.
        assert "specialist_a" in report.plan_deviation_summary
        assert "score_severity" in report.plan_deviation_summary
        assert "insufficient data" in report.plan_deviation_summary

        # LLM explanation used.
        assert report.one_line_explanation
        assert report.one_line_explanation == llm.text
        # Prompt must reference the structured fields, not invent external ones.
        assert len(llm.prompts) == 1
        prompt = llm.prompts[0]
        assert "root_cause_event_id" in prompt
        assert "agent_responsibility_scores" in prompt
        assert "structured fields" in prompt.lower()

        # Persisted.
        persisted = store.get_full_record(TASK_ID).xai_data.accountability_report
        assert persisted is not None
        assert persisted.final_outcome == "MI"
        assert persisted.one_line_explanation == report.one_line_explanation
        assert persisted.most_impactful_tool_call_id == scenario["tool"].tool_call_id

    def test_cf_run_weight_beats_acted_upon_heuristic(self, store, scenario):
        """If a counterfactual run gives the non-acted message a higher delta,
        it should win over acted_upon-only messages."""
        pipeline = MockPipeline({"final_diagnosis": "PE", "confidence": 0.5})
        llm = FakeLLM()
        gen = AccountabilityReportGenerator(store=store, pipeline=pipeline, llm=llm)

        msgs = scenario["messages"]
        # Ensure the counterfactual_runs table exists, then populate deltas for
        # all three messages with m3 explicitly the highest.
        from agentxai.xai.counterfactual_engine import CounterfactualEngine
        CounterfactualEngine(store=store, pipeline=pipeline, task_id=TASK_ID)
        from sqlalchemy import text
        with store._engine.connect() as conn:
            for rid, m, delta in [
                ("cf-m1", msgs["m1"], 0.2),
                ("cf-m2", msgs["m2"], 0.4),
                ("cf-m3", msgs["m3"], 0.9),
            ]:
                conn.execute(
                    text(
                        "INSERT INTO counterfactual_runs "
                        "(run_id, task_id, perturbation_type, target_id, "
                        " baseline_value_json, original_outcome_json, "
                        " perturbed_outcome_json, outcome_delta) "
                        "VALUES (:r, :t, 'message', :m, '{}', '{}', '{}', :d)"
                    ),
                    {"r": rid, "t": TASK_ID, "m": m.message_id, "d": delta},
                )
            conn.commit()

        report = gen.generate(TASK_ID)
        assert report.most_influential_message_id == msgs["m3"].message_id

    def test_no_pipeline_distributes_responsibility_equally(self, store, scenario):
        llm = FakeLLM()
        gen = AccountabilityReportGenerator(store=store, pipeline=None, llm=llm)
        report = gen.generate(TASK_ID)
        # Still populated: 1/N across detected specialists.
        assert sum(report.agent_responsibility_scores.values()) == pytest.approx(1.0)
        assert set(report.agent_responsibility_scores) == {"specialist_a", "specialist_b"}

    def test_no_llm_falls_back_to_templated_sentence(self, store, scenario):
        pipeline = MockPipeline({"final_diagnosis": "PE", "confidence": 0.5})
        gen = AccountabilityReportGenerator(store=store, pipeline=pipeline, llm=None)
        # Force the optional ChatGoogleGenerativeAI path off regardless of env.
        gen.llm = None
        report = gen.generate(TASK_ID)
        assert report.one_line_explanation
        assert "MI" in report.one_line_explanation

    def test_explicit_specialist_list_respected(self, store, scenario):
        pipeline = MockPipeline({"final_diagnosis": "PE", "confidence": 0.5})
        gen = AccountabilityReportGenerator(
            store=store, pipeline=pipeline, llm=FakeLLM(),
            specialist_agents=["specialist_a"],
        )
        report = gen.generate(TASK_ID)
        assert set(report.agent_responsibility_scores) == {"specialist_a"}
        assert report.agent_responsibility_scores["specialist_a"] == pytest.approx(1.0)
