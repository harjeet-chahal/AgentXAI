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
    _AGGREGATOR_ACTIONS,
    _RESP_WEIGHTS,
    _agent_causal_centrality,
    _agent_memory_substance,
    _agent_message_efficacy,
    _agent_tool_impact,
    _agent_usefulness,
    _combine_signals,
    _compute_responsibility_signals,
    _deviation_summary,
    _fallback_explanation,
    _is_aggregator_node,
    _is_substantive,
    _normalize_to_one,
    _select_root_cause,
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
        # No xai supplied — fallback still mentions the top agent and must
        # not leak any raw UUIDs into the sentence.
        s = _fallback_explanation(r)
        assert "specialist_a" in s
        assert "MI" in s
        assert "correct" in s
        assert "e2" not in s  # raw event_id must not leak

    def test_fallback_explanation_credits_tied_agents(self):
        from agentxai.data.schemas import AccountabilityReport
        # Exact 0.50 / 0.50 tie — neither agent should be singled out.
        r = AccountabilityReport(
            final_outcome="GBC", outcome_correct=True,
            agent_responsibility_scores={"specialist_a": 0.50, "specialist_b": 0.50},
            root_cause_event_id="e1", causal_chain=["e1"],
        )
        s = _fallback_explanation(r)
        # Both agents appear, joined as a shared/tied attribution.
        assert "specialist_a" in s
        assert "specialist_b" in s
        # And the language should reflect the tie, not pick one.
        assert "shared" in s.lower() or "tied" in s.lower() or " and " in s

    def test_fallback_explanation_singles_out_clear_winner(self):
        from agentxai.data.schemas import AccountabilityReport
        # 0.97 vs 0.03 — clearly not tied; only specialist_a should be named.
        r = AccountabilityReport(
            final_outcome="DT", outcome_correct=True,
            agent_responsibility_scores={"specialist_a": 0.97, "specialist_b": 0.03},
            root_cause_event_id="e1", causal_chain=["e1"],
        )
        s = _fallback_explanation(r)
        assert "specialist_a" in s
        # specialist_b shouldn't be credited at all when this lopsided.
        assert "specialist_b" not in s

    def test_fallback_explanation_resolves_uuids_when_xai_provided(self):
        from agentxai.data.schemas import (
            AccountabilityReport,
            XAIData,
        )
        ev = TrajectoryEvent(
            event_id="e2", agent_id="specialist_a",
            event_type="tool_call", action="symptom_lookup",
        )
        tc = ToolUseEvent(
            tool_name="symptom_lookup",
            called_by="specialist_a",
            inputs={}, outputs={}, duration_ms=10.0,
        )
        tc.downstream_impact_score = 0.82
        xai = XAIData(trajectory=[ev], tool_calls=[tc])

        r = AccountabilityReport(
            final_outcome="MI", outcome_correct=True,
            agent_responsibility_scores={"specialist_a": 0.7, "specialist_b": 0.3},
            root_cause_event_id="e2",
            most_impactful_tool_call_id=tc.tool_call_id,
            causal_chain=["e2"],
        )
        s = _fallback_explanation(r, xai)
        # Tool surfaces by name, root cause by event_type + agent — never by UUID.
        assert "symptom_lookup" in s
        assert tc.tool_call_id not in s
        assert "tool_call" in s
        assert "specialist_a" in s


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

        # Composite scoring: specialist_a (high-impact tool, acted-upon
        # message, substantive memory) must clearly outrank specialist_b
        # (no tools, ignored message, single substantive write) — even
        # though both perturbations flip the diagnosis identically.
        assert set(report.agent_responsibility_scores) == {"specialist_a", "specialist_b"}
        assert sum(report.agent_responsibility_scores.values()) == pytest.approx(1.0)
        a = report.agent_responsibility_scores["specialist_a"]
        b = report.agent_responsibility_scores["specialist_b"]
        assert a > b
        assert a > 0.55  # not a 0.5/0.5 tie anymore
        assert b < 0.45

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
        # Prompt must reference the resolved structured fields and explicitly
        # forbid leaking raw UUIDs into the explanation.
        assert len(llm.prompts) == 1
        prompt = llm.prompts[0]
        assert "root_cause_event" in prompt
        assert "agent_responsibility_scores" in prompt
        assert "most_impactful_tool_call" in prompt
        assert "structured fields" in prompt.lower()
        # The resolver should embed tool_name, agent_id, etc. — not raw IDs.
        assert "tool_name" in prompt
        # And the rules must include the no-UUID instruction.
        assert "uuid" in prompt.lower() or "tool_call_ids" in prompt.lower()

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

    def test_no_pipeline_uses_non_counterfactual_signals(self, store, scenario):
        """
        Without a pipeline we cannot run counterfactual perturbations, but
        the other five signals (tool impact, message efficacy, memory
        substance, usefulness, causal centrality) still differentiate the
        specialists. specialist_a should still outrank specialist_b.
        """
        llm = FakeLLM()
        gen = AccountabilityReportGenerator(store=store, pipeline=None, llm=llm)
        report = gen.generate(TASK_ID)

        assert sum(report.agent_responsibility_scores.values()) == pytest.approx(1.0)
        assert set(report.agent_responsibility_scores) == {"specialist_a", "specialist_b"}
        # specialist_a: tool_impact=0.9, message_efficacy=1.0, substance=1.0
        # specialist_b: tool_impact=0,   message_efficacy=0.2, substance=1.0
        a = report.agent_responsibility_scores["specialist_a"]
        b = report.agent_responsibility_scores["specialist_b"]
        assert a > b

    def test_all_zero_signals_falls_back_to_equal_split(self, store):
        """
        When pipeline is None and no specialist has any contributing signal
        (no tools, no messages, no memory), `_normalize_to_one` falls back
        to an equal 1/N split so the field stays populated for consumers.
        """
        # Fresh task with two specialists in trajectory but nothing else.
        s = store
        from agentxai.data.schemas import AgentXAIRecord, TrajectoryEvent
        s.save_task(AgentXAIRecord(
            task_id="ZERO-TASK",
            source="test",
            input={}, ground_truth={},
            system_output={"final_diagnosis": "X", "confidence": 0.0, "correct": False},
        ))
        for ag in ("specialist_a", "specialist_b"):
            s.save_event("ZERO-TASK", TrajectoryEvent(
                agent_id=ag, event_type="agent_action", action="noop", timestamp=1.0,
            ))
        gen = AccountabilityReportGenerator(store=s, pipeline=None, llm=FakeLLM())
        report = gen.generate("ZERO-TASK")
        assert report.agent_responsibility_scores == {
            "specialist_a": pytest.approx(0.5),
            "specialist_b": pytest.approx(0.5),
        }

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


# ---------------------------------------------------------------------------
# Bug-fix scenario: empty agent must not get equal share
# ---------------------------------------------------------------------------

@pytest.fixture()
def imbalanced_store_and_ids(store: TrajectoryStore):
    """
    Build a scenario that reproduces the original bug:

      * specialist_a contributed nothing of substance — its only memory
        write is `confidence=0` and `top_conditions=[]`, its message was
        ignored by the synthesizer, and it called no tools.

      * specialist_b did real work — high-impact tool, acted-upon
        finding message, substantive memory write.

    Under the old single-signal scoring both agents flip the diagnosis
    when their memory is zeroed (the MockPipeline returns "PE" regardless),
    so each got 0.5. Under the new composite scoring, specialist_a's empty
    contribution must drag its share well below specialist_b's.
    """
    s = store
    task_id = "BUG-IMBALANCE-001"
    s.save_task(AgentXAIRecord(
        task_id=task_id,
        source="test",
        input={"patient_case": "edge case"},
        ground_truth={"correct_answer": "B"},
        system_output={"final_diagnosis": "MI", "confidence": 0.9, "correct": True},
    ))

    # Trajectory — three agents.
    events = [
        _mk_event("orchestrator", "plan",            1.0, "triage"),
        _mk_event("specialist_a", "agent_action",    2.0, "diagnose"),
        _mk_event("specialist_b", "tool_start",      3.0, "symptom_lookup"),
        _mk_event("specialist_b", "agent_action",    4.0, "diagnose"),
        _mk_event("synthesizer",  "final_diagnosis", 5.0, "MI"),
    ]
    for e in events:
        s.save_event(task_id, e)
    e_orch, e_a, e_b_tool, e_b_act, e_term = events

    # specialist_a — empty contribution.
    s.save_memory_diff(task_id, MemoryDiff(
        agent_id="specialist_a", operation="write", key="top_conditions",
        value_before=None, value_after=[], triggered_by_event_id=e_a.event_id,
    ))
    s.save_memory_diff(task_id, MemoryDiff(
        agent_id="specialist_a", operation="write", key="confidence",
        value_before=None, value_after=0.0, triggered_by_event_id=e_a.event_id,
    ))
    s.save_message(task_id, AgentMessage(
        sender="specialist_a", receiver="synthesizer", timestamp=2.5,
        message_type="finding", content={"dx": None}, acted_upon=False,
    ))

    # specialist_b — substantive contribution.
    s.save_tool_call(task_id, ToolUseEvent(
        tool_name="symptom_lookup", called_by="specialist_b", timestamp=3.1,
        inputs={"symptom": "chest_pain"}, outputs={"conditions": ["MI"]},
        duration_ms=20.0, downstream_impact_score=0.85,
    ))
    s.save_memory_diff(task_id, MemoryDiff(
        agent_id="specialist_b", operation="write", key="top_conditions",
        value_before=None, value_after=["MI"], triggered_by_event_id=e_b_act.event_id,
    ))
    s.save_memory_diff(task_id, MemoryDiff(
        agent_id="specialist_b", operation="write", key="retrieval_confidence",
        value_before=None, value_after=0.8, triggered_by_event_id=e_b_act.event_id,
    ))
    s.save_message(task_id, AgentMessage(
        sender="specialist_b", receiver="synthesizer", timestamp=4.5,
        message_type="finding", content={"dx": "MI"}, acted_upon=True,
    ))

    return s, task_id


class TestCompositeResponsibility:
    """End-to-end coverage of the redesigned responsibility scoring."""

    def test_empty_agent_gets_low_responsibility(self, imbalanced_store_and_ids):
        """
        Bug-fix regression test: an agent with empty memory and an ignored
        message must NOT receive 0.5 just because the Synthesizer's prompt
        structurally read its (empty) memory and zeroing it flipped the dx.
        """
        s, task_id = imbalanced_store_and_ids
        # Pipeline returns a different dx → both perturbations flip outcome,
        # so the counterfactual signal is identical (1.0) for both agents.
        pipeline = MockPipeline({"final_diagnosis": "PE", "confidence": 0.3})
        gen = AccountabilityReportGenerator(store=s, pipeline=pipeline, llm=FakeLLM())
        report = gen.generate(task_id)

        a = report.agent_responsibility_scores["specialist_a"]
        b = report.agent_responsibility_scores["specialist_b"]
        # Empty agent shouldn't even crack one third of total responsibility.
        assert a < 0.34, f"empty specialist_a still got {a:.3f}"
        # Substantive agent dominates.
        assert b > 0.66, f"substantive specialist_b only got {b:.3f}"

    def test_high_impact_agent_gets_high_responsibility(self, imbalanced_store_and_ids):
        """The substantive specialist outranks the empty one by a wide margin."""
        s, task_id = imbalanced_store_and_ids
        pipeline = MockPipeline({"final_diagnosis": "PE", "confidence": 0.3})
        gen = AccountabilityReportGenerator(store=s, pipeline=pipeline, llm=FakeLLM())
        report = gen.generate(task_id)

        a = report.agent_responsibility_scores["specialist_a"]
        b = report.agent_responsibility_scores["specialist_b"]
        assert b > a
        # b should win by at least 2x — a meaningful margin, not a near-tie.
        assert b >= 2 * a, f"specialist_b ({b:.3f}) did not dominate specialist_a ({a:.3f})"

    def test_responsibility_normalized_to_one(self, imbalanced_store_and_ids):
        """Composite scores always sum to exactly 1.0 (within float tolerance)."""
        s, task_id = imbalanced_store_and_ids
        pipeline = MockPipeline({"final_diagnosis": "PE", "confidence": 0.3})
        gen = AccountabilityReportGenerator(store=s, pipeline=pipeline, llm=FakeLLM())
        report = gen.generate(task_id)

        total = sum(report.agent_responsibility_scores.values())
        assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Per-signal helper unit tests
# ---------------------------------------------------------------------------

class TestPerSignalHelpers:
    def test_resp_weights_sum_to_one(self):
        # Combining weights must form a convex combination so the unnormalized
        # score lands in [0, 1] when every signal is in [0, 1].
        assert sum(_RESP_WEIGHTS.values()) == pytest.approx(1.0)

    def test_is_substantive_treats_emptiness_as_uninformative(self):
        for empty in (None, False, 0, 0.0, "", b"", [], (), {}, set(), frozenset()):
            assert not _is_substantive(empty), f"{empty!r} should be uninformative"
        for full in (True, 1, -1, 0.5, "x", b"x", [0], (0,), {"k": "v"}, {1}):
            assert _is_substantive(full), f"{full!r} should be substantive"

    def test_agent_tool_impact_uses_max(self):
        tools = [
            ToolUseEvent(tool_name="a", called_by="alpha", downstream_impact_score=0.1),
            ToolUseEvent(tool_name="b", called_by="alpha", downstream_impact_score=0.7),
            ToolUseEvent(tool_name="c", called_by="beta",  downstream_impact_score=0.9),
        ]
        assert _agent_tool_impact("alpha", tools) == pytest.approx(0.7)
        assert _agent_tool_impact("beta",  tools) == pytest.approx(0.9)
        assert _agent_tool_impact("gamma", tools) == 0.0

    def test_agent_message_efficacy_prefers_cf_delta_over_heuristic(self):
        msgs = [
            AgentMessage(message_id="m1", sender="alpha", acted_upon=True),
            AgentMessage(message_id="m2", sender="alpha", acted_upon=False),
        ]
        # No cf data → heuristic: max(1.0, 0.2) = 1.0
        assert _agent_message_efficacy("alpha", msgs, {}) == pytest.approx(1.0)
        # cf data overrides heuristic — even on the acted-upon message.
        assert _agent_message_efficacy("alpha", msgs, {"m1": 0.2, "m2": 0.9}) == pytest.approx(0.9)
        # No messages from this agent.
        assert _agent_message_efficacy("ghost", msgs, {}) == 0.0

    def test_agent_message_efficacy_low_when_only_message_ignored(self):
        msgs = [AgentMessage(sender="alpha", acted_upon=False)]
        assert _agent_message_efficacy("alpha", msgs, {}) == pytest.approx(0.2)

    def test_agent_memory_substance_fraction_of_substantive_writes(self):
        diffs = [
            MemoryDiff(agent_id="alpha", operation="write", key="x", value_after=[]),
            MemoryDiff(agent_id="alpha", operation="write", key="y", value_after={"k": 1}),
            MemoryDiff(agent_id="alpha", operation="write", key="z", value_after=None),
            # Reads don't count.
            MemoryDiff(agent_id="alpha", operation="read",  key="y", value_after={"k": 1}),
        ]
        assert _agent_memory_substance("alpha", diffs) == pytest.approx(1 / 3)
        assert _agent_memory_substance("none",  diffs) == 0.0

    def test_agent_usefulness_picks_largest_known_key(self):
        diffs = [
            MemoryDiff(agent_id="alpha", operation="write", key="confidence",
                       value_after=0.4, timestamp=1.0),
            MemoryDiff(agent_id="alpha", operation="write", key="retrieval_confidence",
                       value_after=0.9, timestamp=2.0),
            MemoryDiff(agent_id="alpha", operation="write", key="severity_score",
                       value_after="bad-string", timestamp=3.0),
        ]
        # 0.9 wins over 0.4; the bad string is ignored, not raised.
        assert _agent_usefulness("alpha", diffs) == pytest.approx(0.9)

    def test_agent_usefulness_returns_zero_when_no_known_key(self):
        diffs = [MemoryDiff(agent_id="alpha", operation="write", key="other", value_after=1.0)]
        assert _agent_usefulness("alpha", diffs) == 0.0

    def test_agent_causal_centrality_handles_missing_graph(self):
        assert _agent_causal_centrality("alpha", None, "term") == 0.0

    def test_agent_causal_centrality_sums_outgoing_weight_to_terminal(self):
        import networkx as nx
        g: nx.DiGraph = nx.DiGraph()
        g.add_node("e1", agent_id="alpha")
        g.add_node("e2", agent_id="beta")
        g.add_node("term", agent_id="synthesizer")
        g.add_edge("e1", "term", weight=0.8)
        g.add_edge("e2", "term", weight=0.3)
        assert _agent_causal_centrality("alpha", g, "term") == pytest.approx(0.8)
        assert _agent_causal_centrality("beta",  g, "term") == pytest.approx(0.3)

    def test_combine_signals_weighted_sum(self):
        # All-ones signal vector → score equals the sum of weights → 1.0.
        all_ones = {k: 1.0 for k in _RESP_WEIGHTS}
        assert _combine_signals(all_ones) == pytest.approx(1.0)
        # Only the counterfactual signal is on → score equals its weight.
        only_cf = {k: 0.0 for k in _RESP_WEIGHTS}
        only_cf["counterfactual"] = 1.0
        assert _combine_signals(only_cf) == pytest.approx(_RESP_WEIGHTS["counterfactual"])

    def test_compute_responsibility_signals_all_keys_present(self):
        from agentxai.data.schemas import XAIData
        xai = XAIData()
        sig = _compute_responsibility_signals(
            "alpha",
            xai=xai,
            cf_outcome_delta=0.42,
            cf_message_deltas={},
            graph=None,
            terminal_id="",
        )
        # Every weighted signal name must be present so `_combine_signals` is total.
        assert set(sig) == set(_RESP_WEIGHTS)
        # cf is preserved verbatim (clamped); the rest are 0 with empty XAI data.
        assert sig["counterfactual"] == pytest.approx(0.42)
        assert sig["tool_impact"] == 0.0
        assert sig["message_efficacy"] == 0.0
        assert sig["memory_used"] == 0.0
        assert sig["usefulness"] == 0.0
        assert sig["causal_centrality"] == 0.0


# ---------------------------------------------------------------------------
# Root-cause selection
# ---------------------------------------------------------------------------

def _mk_node(graph, event_id, *, agent_id="", action="", event_type="action",
             timestamp=0.0):
    """Add a graph node mirroring the attrs CausalDAGBuilder produces."""
    graph.add_node(
        event_id,
        agent_id=agent_id,
        event_type=event_type,
        action=action,
        timestamp=timestamp,
    )


class TestAggregatorFilter:
    def test_known_aggregator_actions_match(self):
        for action in _AGGREGATOR_ACTIONS:
            assert _is_aggregator_node({"action": action})
        # Case-insensitive.
        assert _is_aggregator_node({"action": "Read_Specialist_Memories"})

    def test_aggregator_prefix_matching(self):
        for action in (
            "route_to_specialist_a",
            "route_to_specialist_b",
            "handoff_to_synthesizer",
            "dispatch_specialists",
        ):
            assert _is_aggregator_node({"action": action})

    def test_plan_event_type_is_aggregator(self):
        assert _is_aggregator_node({"event_type": "plan", "action": ""})
        assert _is_aggregator_node({"event_type": "routing"})

    def test_real_decision_actions_are_not_aggregators(self):
        for action in (
            "pubmed_search",
            "symptom_lookup",
            "guideline_lookup",
            "summarize_findings",
            "synthesize_diagnosis",
            "score_severity",
        ):
            assert not _is_aggregator_node(
                {"action": action, "event_type": "tool_call"}
            )


class TestRootCauseSelection:
    """
    Headline: with a synthetic graph in which `pubmed_search` causes
    `synthesize_diagnosis` and a downstream `read_specialist_memories`
    aggregator sits one hop from the terminal, the selector must pick
    `pubmed_search` — not the aggregator.
    """

    def _build_synthetic_xai_and_graph(self):
        """
        Trajectory:
            t=1  e_search   specialist_b   tool_call   pubmed_search
            t=2  e_summary  specialist_b   action      summarize_findings  (writes mem, sends acted msg)
            t=3  e_read     synthesizer    action      read_specialist_memories  (aggregator)
            t=4  e_term     synthesizer    final       synthesize_diagnosis

        Edges (mirror the CausalDAGBuilder output):
            e_search  -> e_summary   weight 0.85  (tool impact)
            e_summary -> e_read      weight 1.00  (acted-upon message)
            e_read    -> e_term      weight 0.30  (within-agent temporal)
        """
        e_search   = "evt-search"
        e_summary  = "evt-summary"
        e_read     = "evt-read"
        e_term     = "evt-term"

        events = [
            TrajectoryEvent(event_id=e_search,  timestamp=1.0,
                            agent_id="specialist_b", event_type="tool_call",
                            action="pubmed_search"),
            TrajectoryEvent(event_id=e_summary, timestamp=2.0,
                            agent_id="specialist_b", event_type="agent_action",
                            action="summarize_findings"),
            TrajectoryEvent(event_id=e_read,    timestamp=3.0,
                            agent_id="synthesizer",  event_type="agent_action",
                            action="read_specialist_memories"),
            TrajectoryEvent(event_id=e_term,    timestamp=4.0,
                            agent_id="synthesizer",  event_type="final_diagnosis",
                            action="synthesize_diagnosis"),
        ]

        tool = ToolUseEvent(
            tool_name="pubmed_search", called_by="specialist_b", timestamp=1.0,
            inputs={"q": "MI"}, outputs={"docs": []},
            duration_ms=10.0, downstream_impact_score=0.85,
        )
        msg = AgentMessage(
            sender="specialist_b", receiver="synthesizer", timestamp=2.5,
            message_type="finding", content={"dx": "MI"}, acted_upon=True,
        )
        diff = MemoryDiff(
            agent_id="specialist_b", operation="write", key="top_evidence",
            value_before=None, value_after=["MI per Harrison 2018"],
            triggered_by_event_id=e_summary,
        )

        from agentxai.data.schemas import XAIData
        xai = XAIData(
            trajectory=events,
            tool_calls=[tool],
            messages=[msg],
            memory_diffs=[diff],
        )

        import networkx as nx
        g: nx.DiGraph = nx.DiGraph()
        for e in events:
            _mk_node(g, e.event_id, agent_id=e.agent_id, action=e.action,
                     event_type=e.event_type, timestamp=e.timestamp)
        g.add_edge(e_search,  e_summary, weight=0.85, causal_type="direct")
        g.add_edge(e_summary, e_read,    weight=1.00, causal_type="direct")
        g.add_edge(e_read,    e_term,    weight=0.30, causal_type="contributory")

        return xai, g, {
            "search": e_search, "summary": e_summary,
            "read": e_read, "term": e_term,
        }

    def test_pubmed_search_beats_read_specialist_memories(self):
        """
        The exact bug from the user report: under the old graph-weight-only
        selector, `read_specialist_memories` would win because its
        outgoing edge to the terminal carries weight 1.0 vs `pubmed_search`'s
        chain weight further upstream. New selector must filter the
        aggregator and pick `pubmed_search`.
        """
        xai, g, ids = self._build_synthetic_xai_and_graph()
        root_id, reason = _select_root_cause(g, ids["term"], xai)

        assert root_id == ids["search"], (
            f"Expected pubmed_search ({ids['search']}) but got "
            f"{root_id} with reason {reason!r}"
        )
        assert ids["read"] not in {root_id}
        # Reason should name the action, the agent, and the tool-impact bonus.
        assert "pubmed_search" in reason
        assert "specialist_b" in reason
        assert "high-impact tool" in reason

    def test_read_specialist_memories_excluded_from_candidates(self):
        """The aggregator must never be selected, even with no other signals."""
        xai, g, ids = self._build_synthetic_xai_and_graph()
        # Strip the tool / message / memory bonuses — read_* now has the
        # *only* meaningful outgoing weight. It still must not be picked.
        xai.tool_calls = []
        xai.messages = []
        xai.memory_diffs = []

        root_id, reason = _select_root_cause(g, ids["term"], xai)
        assert root_id != ids["read"]
        # Must fall back to a non-aggregator ancestor (search or summary).
        assert root_id in {ids["search"], ids["summary"]}

    def test_acted_upon_message_anchor_event_gets_bonus(self):
        """
        An ancestor that produced an acted-upon outgoing message gets a
        positive bonus in scoring. Verified by stripping the tool bonus
        and checking summary (which sent the acted-upon message) outranks
        search.
        """
        xai, g, ids = self._build_synthetic_xai_and_graph()
        # Remove the tool so search no longer has its tool-impact bonus —
        # then summary's acted-upon message bonus + substantive memory
        # write should dominate.
        xai.tool_calls = []
        # Also flatten the search→summary graph weight so neither has a
        # base advantage.
        g[ids["search"]][ids["summary"]]["weight"] = 0.0

        root_id, reason = _select_root_cause(g, ids["term"], xai)
        assert root_id == ids["summary"]
        assert "acted-upon message" in reason
        assert "substantive memory write" in reason

    def test_substantive_memory_alone_can_win(self):
        """An event that triggered a substantive memory write outranks a bare ancestor."""
        import networkx as nx
        from agentxai.data.schemas import XAIData
        g: nx.DiGraph = nx.DiGraph()
        _mk_node(g, "e_bare",  agent_id="specialist_a", action="extract_symptoms",
                 timestamp=1.0)
        _mk_node(g, "e_write", agent_id="specialist_a", action="lookup_conditions",
                 timestamp=2.0)
        _mk_node(g, "e_term",  agent_id="synthesizer", event_type="final_diagnosis",
                 timestamp=3.0)
        g.add_edge("e_bare",  "e_term", weight=0.30, causal_type="contributory")
        g.add_edge("e_write", "e_term", weight=0.30, causal_type="contributory")

        xai = XAIData(
            trajectory=[
                TrajectoryEvent(event_id="e_bare",  timestamp=1.0,
                                agent_id="specialist_a", action="extract_symptoms"),
                TrajectoryEvent(event_id="e_write", timestamp=2.0,
                                agent_id="specialist_a", action="lookup_conditions"),
                TrajectoryEvent(event_id="e_term",  timestamp=3.0,
                                agent_id="synthesizer"),
            ],
            memory_diffs=[
                MemoryDiff(agent_id="specialist_a", operation="write",
                           key="top_conditions", value_after=["MI"],
                           triggered_by_event_id="e_write"),
            ],
        )
        root_id, reason = _select_root_cause(g, "e_term", xai)
        assert root_id == "e_write"
        assert "substantive memory write" in reason

    def test_returns_empty_when_terminal_missing(self):
        import networkx as nx
        from agentxai.data.schemas import XAIData
        g: nx.DiGraph = nx.DiGraph()
        _mk_node(g, "e1")
        assert _select_root_cause(g, "", XAIData()) == ("", "")
        assert _select_root_cause(g, "ghost-id", XAIData()) == ("", "")

    def test_returns_empty_when_terminal_has_no_ancestors(self):
        import networkx as nx
        from agentxai.data.schemas import XAIData
        g: nx.DiGraph = nx.DiGraph()
        _mk_node(g, "term")
        assert _select_root_cause(g, "term", XAIData()) == ("", "")

    def test_falls_back_when_all_ancestors_are_aggregators(self):
        """
        If literally every ancestor is an aggregator, return the best of
        them anyway (with a marker in the reason) rather than empty.
        """
        import networkx as nx
        from agentxai.data.schemas import XAIData
        g: nx.DiGraph = nx.DiGraph()
        _mk_node(g, "r1", agent_id="orchestrator", action="route_to_specialist_a",
                 timestamp=1.0)
        _mk_node(g, "r2", agent_id="orchestrator", action="handoff_to_synthesizer",
                 timestamp=2.0)
        _mk_node(g, "term", agent_id="synthesizer", event_type="final_diagnosis",
                 timestamp=3.0)
        g.add_edge("r1", "r2",   weight=0.5, causal_type="direct")
        g.add_edge("r2", "term", weight=0.5, causal_type="direct")

        root_id, reason = _select_root_cause(g, "term", XAIData(
            trajectory=[
                TrajectoryEvent(event_id="r1", timestamp=1.0,
                                agent_id="orchestrator", action="route_to_specialist_a"),
                TrajectoryEvent(event_id="r2", timestamp=2.0,
                                agent_id="orchestrator", action="handoff_to_synthesizer"),
                TrajectoryEvent(event_id="term", timestamp=3.0,
                                agent_id="synthesizer"),
            ],
        ))
        assert root_id in {"r1", "r2"}
        assert "no non-aggregator" in reason

    def test_upstream_preference_breaks_near_ties(self):
        """
        Two events with identical raw scores: earlier one wins thanks to the
        upstream-discount factor.
        """
        import networkx as nx
        from agentxai.data.schemas import XAIData
        g: nx.DiGraph = nx.DiGraph()
        _mk_node(g, "early", agent_id="specialist_a", action="lookup_conditions",
                 timestamp=1.0)
        _mk_node(g, "late",  agent_id="specialist_a", action="score_severity",
                 timestamp=10.0)
        _mk_node(g, "term",  agent_id="synthesizer", event_type="final_diagnosis",
                 timestamp=11.0)
        # Identical outgoing weights → tie-break must prefer the earlier event.
        g.add_edge("early", "term", weight=0.5, causal_type="direct")
        g.add_edge("late",  "term", weight=0.5, causal_type="direct")

        root_id, _ = _select_root_cause(g, "term", XAIData(
            trajectory=[
                TrajectoryEvent(event_id="early", timestamp=1.0,
                                agent_id="specialist_a", action="lookup_conditions"),
                TrajectoryEvent(event_id="late",  timestamp=10.0,
                                agent_id="specialist_a", action="score_severity"),
                TrajectoryEvent(event_id="term",  timestamp=11.0,
                                agent_id="synthesizer"),
            ],
        ))
        assert root_id == "early"


class TestRootCauseInReport:
    """End-to-end integration: generate() populates the new field correctly."""

    def test_generate_populates_root_cause_reason(self, store, scenario):
        pipeline = MockPipeline({"final_diagnosis": "PE", "confidence": 0.5})
        gen = AccountabilityReportGenerator(
            store=store, pipeline=pipeline,
            llm=FakeLLM("Specialist A's symptom_lookup tool drove the diagnosis."),
        )
        report = gen.generate(TASK_ID)

        assert report.root_cause_reason, "root_cause_reason should be populated"
        # Should mention the chosen event's agent, not be a raw UUID.
        assert report.root_cause_event_id not in report.root_cause_reason
        assert any(
            phrase in report.root_cause_reason
            for phrase in ("specialist_a", "specialist_b", "orchestrator")
        )

        # Persisted via the store with the new column.
        persisted = store.get_full_record(TASK_ID).xai_data.accountability_report
        assert persisted is not None
        assert persisted.root_cause_reason == report.root_cause_reason

    def test_fallback_explanation_uses_root_cause_reason(self):
        from agentxai.data.schemas import AccountabilityReport, XAIData
        ev = TrajectoryEvent(
            event_id="e2", agent_id="specialist_b",
            event_type="tool_call", action="pubmed_search",
        )
        r = AccountabilityReport(
            final_outcome="MI", outcome_correct=True,
            agent_responsibility_scores={"specialist_b": 1.0},
            root_cause_event_id="e2",
            root_cause_reason="pubmed_search from specialist_b: high-impact tool (0.85)",
            causal_chain=["e2"],
        )
        s = _fallback_explanation(r, XAIData(trajectory=[ev]))
        # Reason verbatim, not the older "<event_type> from <agent>" form.
        assert "rooted in pubmed_search from specialist_b: high-impact tool" in s
        # Old phrasing must NOT also be appended (no double-rooting).
        assert s.count("rooted in") == 1


# ---------------------------------------------------------------------------
# Memory-usage attribution
# ---------------------------------------------------------------------------

class TestMemoryUsageHeuristic:
    """Direct unit tests for the substring-match attribution heuristic."""

    def test_extract_value_tokens_drops_uninformative_leaves(self):
        from agentxai.xai.memory_usage import extract_value_tokens

        # Numerics, booleans, None, and short strings are all skipped.
        assert extract_value_tokens(0) == []
        assert extract_value_tokens(0.5) == []
        assert extract_value_tokens(True) == []
        assert extract_value_tokens(None) == []
        assert extract_value_tokens("hi") == []   # < _MIN_TOKEN_LEN

        # Strings are tokenized at word boundaries — paraphrased rationales
        # ("Myocardial-infarction-like presentation") still match the
        # individual concept words.
        toks = extract_value_tokens(["Myocardial infarction"])
        assert "Myocardial" in toks
        assert "infarction" in toks

        # Dict VALUES are extracted; dict KEYS are dropped to avoid matching
        # schema labels like "first_line".
        toks = extract_value_tokens({"first_line": "Ceftriaxone 500 mg"})
        assert "Ceftriaxone" in toks
        assert "first_line" not in toks
        # Pure-numeric runs ("500", "mg" is < min len) are filtered too.
        assert "500" not in toks

        # Stoplist drops common glue words even when long enough.
        toks = extract_value_tokens(["the patient was admitted with chest pain"])
        assert "the" not in [t.lower() for t in toks]
        assert "patient" in toks  # domain-y words are kept

        # Tokens are de-duplicated case-insensitively.
        toks = extract_value_tokens(["MI", "MI again", "mi confirmed"])
        # "MI" is < min length so it gets filtered, but "again" / "confirmed"
        # appear once each.
        lowered = [t.lower() for t in toks]
        assert lowered.count("again") == 1
        assert lowered.count("confirmed") == 1

    def test_attribute_memory_usage_basic(self):
        from agentxai.xai.memory_usage import attribute_memory_usage

        diffs = [
            MemoryDiff(agent_id="specialist_b", operation="write",
                       key="top_evidence",
                       value_after=["Ceftriaxone is first-line for gonorrhea"]),
            MemoryDiff(agent_id="specialist_b", operation="write",
                       key="confidence",
                       value_after=0.8),
            MemoryDiff(agent_id="specialist_a", operation="write",
                       key="top_conditions",
                       value_after=[]),
        ]
        rationale = (
            "Gram-negative diplococci with a Ceftriaxone-responsive course "
            "fits gonorrhea — the top_evidence supports the diagnosis."
        )
        usage = attribute_memory_usage(rationale, diffs)

        # Indexed lookup so we don't depend on sort order beyond influence.
        by_key = {(u.agent_id, u.key): u for u in usage}

        # B's top_evidence value is cited verbatim → influence > 0, used.
        ev = by_key[("specialist_b", "top_evidence")]
        assert ev.used_in_final_answer is True
        assert ev.influence_score > 0.0
        assert ev.read_by == ["synthesizer"]

        # B's `confidence=0.8` has no string tokens → influence=0, NOT used.
        conf = by_key[("specialist_b", "confidence")]
        assert conf.used_in_final_answer is False
        assert conf.influence_score == 0.0

        # A's empty list has no tokens → influence=0, NOT used.
        a_top = by_key[("specialist_a", "top_conditions")]
        assert a_top.used_in_final_answer is False
        assert a_top.influence_score == 0.0

    def test_attribute_memory_usage_empty_rationale_returns_zero_influence(self):
        from agentxai.xai.memory_usage import attribute_memory_usage

        diffs = [
            MemoryDiff(agent_id="specialist_b", operation="write",
                       key="top_evidence", value_after=["Ceftriaxone"]),
        ]
        usage = attribute_memory_usage("", diffs)
        assert len(usage) == 1
        assert usage[0].influence_score == 0.0
        assert usage[0].used_in_final_answer is False

    def test_attribute_memory_usage_uses_latest_write_per_key(self):
        from agentxai.xai.memory_usage import attribute_memory_usage

        diffs = [
            MemoryDiff(agent_id="specialist_b", operation="write", key="dx",
                       value_after="Pneumonia", timestamp=1.0),
            # Later write overwrites the first one — only this value should
            # be matched against the rationale.
            MemoryDiff(agent_id="specialist_b", operation="write", key="dx",
                       value_after="Myocardial infarction", timestamp=2.0),
        ]
        rationale = "Findings are most consistent with Myocardial infarction."
        usage = attribute_memory_usage(rationale, diffs)
        assert len(usage) == 1
        assert usage[0].used_in_final_answer is True
        assert usage[0].influence_score == pytest.approx(1.0)

    def test_attribute_memory_usage_owner_filter(self):
        from agentxai.xai.memory_usage import attribute_memory_usage

        diffs = [
            MemoryDiff(agent_id="specialist_a", operation="write",
                       key="top_conditions", value_after=["MI"]),
            MemoryDiff(agent_id="orchestrator", operation="write",
                       key="state", value_after="dispatching"),
        ]
        usage = attribute_memory_usage(
            "MI is the diagnosis", diffs, owner_agents=["specialist_a"],
        )
        # Orchestrator entry filtered out.
        assert len(usage) == 1
        assert usage[0].agent_id == "specialist_a"


class TestMemoryUsedSignal:
    """Per-agent memory_used score driving the responsibility scoring."""

    def test_falls_back_to_substance_when_rationale_empty(self):
        from agentxai.xai.accountability import (
            _agent_memory_substance, _agent_memory_used,
        )
        diffs = [
            MemoryDiff(agent_id="alpha", operation="write", key="x",
                       value_after=["MI"]),
        ]
        # Empty rationale → identical to the structural fallback.
        assert _agent_memory_used("alpha", diffs, "") == _agent_memory_substance("alpha", diffs)

    def test_unused_substantive_memory_scores_zero(self):
        from agentxai.xai.accountability import _agent_memory_used
        diffs = [
            MemoryDiff(agent_id="alpha", operation="write", key="top",
                       value_after=["Pneumonia", "Pulmonary embolism"]),
        ]
        # Rationale doesn't mention either condition → influence=0.
        assert _agent_memory_used(
            "alpha", diffs, "Findings are most consistent with Sepsis."
        ) == 0.0

    def test_cited_memory_scores_above_zero(self):
        from agentxai.xai.accountability import _agent_memory_used
        diffs = [
            MemoryDiff(agent_id="beta", operation="write", key="ev",
                       value_after=["Ceftriaxone is first-line for gonorrhea"]),
        ]
        score = _agent_memory_used(
            "beta", diffs,
            "Per CDC, Ceftriaxone is the first-line antibiotic.",
        )
        assert score > 0.0
        assert score <= 1.0

    def test_empty_writes_not_in_denominator(self):
        from agentxai.xai.accountability import _agent_memory_used
        diffs = [
            MemoryDiff(agent_id="alpha", operation="write", key="empty",
                       value_after=[]),
            MemoryDiff(agent_id="alpha", operation="write", key="zero",
                       value_after=0),
        ]
        # All values uninformative → score is 0, not 0/0.
        assert _agent_memory_used("alpha", diffs, "anything") == 0.0


class TestMemoryUsedInResponsibility:
    """End-to-end: unused memory must not earn responsibility."""

    def _build_store_with_rationale(
        self,
        rationale: str,
        *,
        a_value, b_value,
    ):
        s = TrajectoryStore(db_url="sqlite:///:memory:")
        task_id = "USAGE-TASK"
        s.save_task(AgentXAIRecord(
            task_id=task_id,
            source="test",
            input={"patient_case": "x"},
            ground_truth={},
            system_output={
                "final_diagnosis": "MI",
                "confidence": 0.9,
                "correct": True,
                "rationale": rationale,
            },
        ))
        for ag in ("specialist_a", "specialist_b"):
            s.save_event(task_id, TrajectoryEvent(
                agent_id=ag, event_type="agent_action",
                action="diagnose", timestamp=1.0,
            ))
        # Both specialists have ONE substantive memory write, and otherwise
        # identical signals (no tools, no messages) — so the only thing
        # that should differentiate them is whether their value got cited.
        s.save_memory_diff(task_id, MemoryDiff(
            agent_id="specialist_a", operation="write",
            key="top_conditions", value_after=a_value,
        ))
        s.save_memory_diff(task_id, MemoryDiff(
            agent_id="specialist_b", operation="write",
            key="top_evidence", value_after=b_value,
        ))
        return s, task_id

    def test_specialist_a_unused_memory_does_not_get_high_responsibility(self):
        """
        Bug-fix regression: the user's example. Specialist A wrote some
        content, the Synthesizer "read" it, but the rationale only cites
        Specialist B's evidence. A must score below B.
        """
        s, task_id = self._build_store_with_rationale(
            rationale=(
                "Per the CDC, Ceftriaxone is first-line treatment for "
                "gonorrhea. The retrieved guidelines support this."
            ),
            a_value=["Asymptomatic carrier"],   # not mentioned in rationale
            b_value=["Ceftriaxone is first-line for gonorrhea"],  # cited
        )
        pipeline = MockPipeline({"final_diagnosis": "PE", "confidence": 0.5})
        gen = AccountabilityReportGenerator(
            store=s, pipeline=pipeline, llm=FakeLLM(),
        )
        report = gen.generate(task_id)
        a = report.agent_responsibility_scores["specialist_a"]
        b = report.agent_responsibility_scores["specialist_b"]
        assert b > a
        assert sum(report.agent_responsibility_scores.values()) == pytest.approx(1.0)

        # The MemoryUsage records persist on the report.
        usage_by_agent = {(u.agent_id, u.key): u for u in report.memory_usage}
        assert usage_by_agent[("specialist_a", "top_conditions")].used_in_final_answer is False
        assert usage_by_agent[("specialist_b", "top_evidence")].used_in_final_answer is True
        assert usage_by_agent[("specialist_b", "top_evidence")].influence_score > 0.0

    def test_specialist_b_top_evidence_used_increases_responsibility(self):
        """
        Stricter version: when B's evidence is cited verbatim and A's is
        not cited at all, B's share should clearly dominate (>= 1.5× A).
        """
        s, task_id = self._build_store_with_rationale(
            rationale=(
                "Ceftriaxone is the first-line antibiotic per the CDC's "
                "guidelines, fitting Neisseria gonorrhoeae."
            ),
            a_value=["Heat exhaustion"],         # 0 token matches
            b_value=["Ceftriaxone", "first-line antibiotic", "CDC"],  # all match
        )
        pipeline = MockPipeline({"final_diagnosis": "PE", "confidence": 0.5})
        gen = AccountabilityReportGenerator(
            store=s, pipeline=pipeline, llm=FakeLLM(),
        )
        report = gen.generate(task_id)
        a = report.agent_responsibility_scores["specialist_a"]
        b = report.agent_responsibility_scores["specialist_b"]
        assert b >= 1.5 * a, (
            f"specialist_b ({b:.3f}) should clearly outrank specialist_a "
            f"({a:.3f}) when its memory is cited and A's is ignored"
        )

    def test_memory_usage_persists_through_store_round_trip(self):
        """
        The new `memory_usage_json` column survives `save → get_full_record`,
        rebuilt as MemoryUsage dataclasses on the way out.
        """
        s, task_id = self._build_store_with_rationale(
            rationale="Findings consistent with Myocardial infarction.",
            a_value=["Heat exhaustion"],
            b_value=["Myocardial infarction"],
        )
        pipeline = MockPipeline({"final_diagnosis": "PE", "confidence": 0.5})
        gen = AccountabilityReportGenerator(
            store=s, pipeline=pipeline, llm=FakeLLM(),
        )
        gen.generate(task_id)

        loaded = s.get_full_record(task_id).xai_data.accountability_report
        assert loaded is not None
        assert loaded.memory_usage, "memory_usage should round-trip through SQLite"
        # Rebuilt as dataclasses, not raw dicts.
        from agentxai.data.schemas import MemoryUsage
        assert all(isinstance(u, MemoryUsage) for u in loaded.memory_usage)


# ---------------------------------------------------------------------------
# Question-type priors on responsibility scoring
# ---------------------------------------------------------------------------

class TestQuestionTypePriors:
    """
    The pipeline tags each task with a heuristic question_type. The
    accountability scorer uses it to apply a per-(type, agent) prior
    multiplier, then renormalises. This shrinks Specialist A's share on
    questions where symptom analysis is irrelevant (the user-reported HIV
    confirmatory-test scenario).
    """

    def _build_store_and_task(
        self,
        question_type: str,
    ):
        """
        Identical scenario for both specialists — same tools, messages,
        memory writes — so any difference between question_types comes
        purely from the prior multiplier.
        """
        s = TrajectoryStore(db_url="sqlite:///:memory:")
        task_id = f"PRIOR-{question_type}"
        s.save_task(AgentXAIRecord(
            task_id=task_id,
            source="test",
            input={
                "patient_case": "x",
                "options": {},
                "question_type": question_type,
            },
            ground_truth={},
            system_output={
                "final_diagnosis": "X",
                "confidence": 0.8,
                "correct": True,
                "rationale": "MI",
            },
        ))
        # Same minimal trajectory for both specialists.
        for ag in ("specialist_a", "specialist_b"):
            s.save_event(task_id, TrajectoryEvent(
                agent_id=ag, event_type="agent_action",
                action="diagnose", timestamp=1.0,
            ))
        # Same single substantive memory write per specialist.
        s.save_memory_diff(task_id, MemoryDiff(
            agent_id="specialist_a", operation="write",
            key="top_conditions", value_after=["MI"],
        ))
        s.save_memory_diff(task_id, MemoryDiff(
            agent_id="specialist_b", operation="write",
            key="top_evidence", value_after=["MI"],
        ))
        return s, task_id

    def test_screening_or_test_shrinks_specialist_a_share(self):
        """
        Bug-fix regression: the user's HIV confirmatory-test scenario.
        With identical observable signals between A and B, the
        screening_or_test prior (A=0.5, B=1.0) must give A ≤ 1/3 of the
        total responsibility instead of the would-be 0.5.
        """
        s, task_id = self._build_store_and_task("screening_or_test")
        pipeline = MockPipeline({"final_diagnosis": "Y", "confidence": 0.5})
        gen = AccountabilityReportGenerator(
            store=s, pipeline=pipeline, llm=FakeLLM(),
        )
        report = gen.generate(task_id)
        a = report.agent_responsibility_scores["specialist_a"]
        b = report.agent_responsibility_scores["specialist_b"]
        assert sum(report.agent_responsibility_scores.values()) == pytest.approx(1.0)
        # Prior is 0.5/1.0 → A's pre-normalisation contribution is halved.
        assert a < 0.40, (
            f"specialist_a should be shrunk by the screening prior; got {a:.3f}"
        )
        assert b > 0.60

    def test_diagnosis_prior_is_neutral(self):
        """
        For a classic diagnosis question A and B should be (close to)
        equal when their underlying signals are equal — the prior table
        sets both to 1.0 so renormalisation is the only operator.
        """
        s, task_id = self._build_store_and_task("diagnosis")
        pipeline = MockPipeline({"final_diagnosis": "Y", "confidence": 0.5})
        gen = AccountabilityReportGenerator(
            store=s, pipeline=pipeline, llm=FakeLLM(),
        )
        report = gen.generate(task_id)
        a = report.agent_responsibility_scores["specialist_a"]
        b = report.agent_responsibility_scores["specialist_b"]
        assert a == pytest.approx(b, abs=1e-6)
        assert a == pytest.approx(0.5, abs=1e-6)

    def test_pharmacology_shrinks_specialist_a(self):
        """Same shape as the screening test, this time for pharmacology."""
        s, task_id = self._build_store_and_task("pharmacology")
        pipeline = MockPipeline({"final_diagnosis": "Y", "confidence": 0.5})
        gen = AccountabilityReportGenerator(
            store=s, pipeline=pipeline, llm=FakeLLM(),
        )
        report = gen.generate(task_id)
        a = report.agent_responsibility_scores["specialist_a"]
        b = report.agent_responsibility_scores["specialist_b"]
        # Pharmacology prior: A=0.6, B=1.0 → A's pre-norm score = 0.6× B's →
        # after norm A ≤ 0.6/(0.6+1.0) = 0.375.
        assert a < 0.40
        assert b > 0.60

    def test_unknown_question_type_is_neutral(self):
        """
        Default behavior: when the classifier returned `unknown` (or the
        field is missing entirely on legacy records), priors don't
        multiply — preserves backward-compat scoring.
        """
        s, task_id = self._build_store_and_task("unknown")
        pipeline = MockPipeline({"final_diagnosis": "Y", "confidence": 0.5})
        gen = AccountabilityReportGenerator(
            store=s, pipeline=pipeline, llm=FakeLLM(),
        )
        report = gen.generate(task_id)
        a = report.agent_responsibility_scores["specialist_a"]
        b = report.agent_responsibility_scores["specialist_b"]
        assert a == pytest.approx(b, abs=1e-6)

    def test_missing_question_type_field_treated_as_unknown(self):
        """
        Legacy records (created before the classifier was added) have
        no `question_type` on input. The scorer must treat that as
        unknown (neutral), not crash.
        """
        s = TrajectoryStore(db_url="sqlite:///:memory:")
        task_id = "PRIOR-LEGACY"
        s.save_task(AgentXAIRecord(
            task_id=task_id, source="test",
            input={"patient_case": "x"},   # no question_type key
            ground_truth={},
            system_output={"final_diagnosis": "X", "confidence": 0.8,
                           "correct": True, "rationale": "MI"},
        ))
        for ag in ("specialist_a", "specialist_b"):
            s.save_event(task_id, TrajectoryEvent(
                agent_id=ag, event_type="agent_action",
                action="diagnose", timestamp=1.0,
            ))
        s.save_memory_diff(task_id, MemoryDiff(
            agent_id="specialist_a", operation="write", key="x", value_after=["MI"],
        ))
        s.save_memory_diff(task_id, MemoryDiff(
            agent_id="specialist_b", operation="write", key="x", value_after=["MI"],
        ))

        pipeline = MockPipeline({"final_diagnosis": "Y", "confidence": 0.5})
        gen = AccountabilityReportGenerator(
            store=s, pipeline=pipeline, llm=FakeLLM(),
        )
        # Should not raise; should produce neutral scoring.
        report = gen.generate(task_id)
        a = report.agent_responsibility_scores["specialist_a"]
        b = report.agent_responsibility_scores["specialist_b"]
        assert a == pytest.approx(b, abs=1e-6)
