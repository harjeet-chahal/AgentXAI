"""
Unit tests for the four agents under ``agentxai/agents/``.

Each test wires the agent against a real in-memory TrajectoryStore (so the
Pillar-1..5 logging is exercised end-to-end) but mocks every external
dependency: tool callables and the LLM. After running the agent we assert:

  - the expected memory keys were written
  - the expected finding/handoff messages were sent
  - the plan registered the right number of intended actions
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from agentxai.agents.orchestrator import Orchestrator, _ORCHESTRATOR_PLAN
from agentxai.agents.specialist_a import SpecialistA
from agentxai.agents.specialist_b import SpecialistB
from agentxai.agents.synthesizer import Synthesizer
from agentxai.data.schemas import AgentXAIRecord
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.memory_logger import MemoryLogger
from agentxai.xai.message_logger import MessageLogger
from agentxai.xai.plan_tracker import PlanTracker
from agentxai.xai.trajectory_logger import TrajectoryLogger


TASK_ID = "AGENT-TEST-001"
CASE_TEXT = (
    "A 58-year-old man presents with crushing substernal chest pain radiating "
    "to the left arm, dyspnea, and diaphoresis. ECG shows ST elevations in II, III, aVF."
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store() -> TrajectoryStore:
    s = TrajectoryStore(db_url="sqlite:///:memory:")
    s.save_task(AgentXAIRecord(task_id=TASK_ID, source="test"))
    return s


@pytest.fixture()
def loggers(store: TrajectoryStore) -> Dict[str, Any]:
    """One shared set of loggers so every agent writes to the same task."""
    return {
        "trajectory_logger": TrajectoryLogger(store, TASK_ID),
        # llm=None disables PlanTracker's deviation-explanation LLM call.
        "plan_tracker":      PlanTracker(store, TASK_ID, llm=None),
        "memory_logger":     MemoryLogger(store, TASK_ID),
        "message_logger":    MessageLogger(store, TASK_ID),
    }


def _fake_llm(content: Any) -> MagicMock:
    """A LangChain-style chat LLM whose .invoke() returns a message-like object."""
    fake = MagicMock()
    response = MagicMock()
    response.content = content if isinstance(content, str) else json.dumps(content)
    fake.invoke.return_value = response
    return fake


def _fake_llm_seq(*contents: Any) -> MagicMock:
    """
    LangChain-style chat LLM that returns each ``contents`` entry in turn,
    one per ``.invoke()`` call. Use when a single test path needs different
    LLM payloads for the plan / extraction / tool-gating calls.
    """
    fake = MagicMock()
    responses = []
    for c in contents:
        r = MagicMock()
        r.content = c if isinstance(c, str) else json.dumps(c)
        responses.append(r)
    fake.invoke.side_effect = responses
    return fake


def _messages_in(store: TrajectoryStore) -> List:
    return store.get_full_record(TASK_ID).xai_data.messages


def _plans_in(store: TrajectoryStore) -> List:
    return store.get_full_record(TASK_ID).xai_data.plans


# ---------------------------------------------------------------------------
# Specialist A
# ---------------------------------------------------------------------------

def test_specialist_a_writes_memory_and_sends_finding(store, loggers):
    """LLM extracts symptoms; tools mocked; assert all 4 memory keys + finding."""
    fake_symptoms = ["chest pain", "dyspnea", "diaphoresis"]
    llm = _fake_llm(fake_symptoms)

    # symptom_lookup returns different related_conditions per symptom.
    def fake_symptom_lookup(symptom: str) -> dict:
        table = {
            "chest pain":  [("Acute MI", 0.6), ("Pulmonary embolism", 0.3)],
            "dyspnea":     [("Acute MI", 0.4), ("CHF", 0.4)],
            "diaphoresis": [],   # no hits
        }
        return {"related_conditions": table.get(symptom, []), "source": "medqa_derived"}

    fake_severity = MagicMock(return_value=0.78)

    agent = SpecialistA(
        agent_id="specialist_a",
        symptom_lookup_fn=fake_symptom_lookup,
        severity_scorer_fn=fake_severity,
        orchestrator_id="orchestrator",
        llm=llm,
        **loggers,
    )

    out = agent.run({"patient_case": CASE_TEXT})

    # Return value mirrors memory.
    assert set(out.keys()) == {"symptom_patterns", "severity_score",
                               "top_conditions", "confidence"}

    # Memory written for all four canonical keys.
    mem = loggers["memory_logger"].for_agent("specialist_a")
    assert set(mem.keys()) == {"symptom_patterns", "severity_score",
                               "top_conditions", "confidence"}
    assert mem["symptom_patterns"] == fake_symptoms
    assert mem["severity_score"] == pytest.approx(0.78)
    # 2 of 3 symptoms produced hits → confidence = 2/3.
    assert mem["confidence"] == pytest.approx(round(2 / 3, 4))
    # Top condition should be "Acute MI" (0.6 + 0.4 = 1.0).
    assert mem["top_conditions"][0][0] == "Acute MI"
    assert mem["top_conditions"][0][1] == pytest.approx(1.0)

    # Tools were called the expected number of times. Four LLM calls now:
    # generate_plan, extract_symptoms, _select_lookup_subset (LLM-gated tool
    # selection for symptom_lookup), and _decide_score_severity (LLM-gated
    # tool selection for severity_scorer). The fake LLM returns the same
    # symptom list every time, so the gating helpers fall through to their
    # default "run everything" branches and the underlying tool callables
    # see the same arguments as before.
    assert llm.invoke.call_count == 4
    assert fake_severity.call_count == 1
    fake_severity.assert_called_with(fake_symptoms)

    # Exactly one finding message to the orchestrator.
    msgs = [m for m in _messages_in(store) if m.sender == "specialist_a"]
    assert len(msgs) == 1
    msg = msgs[0]
    assert msg.receiver == "orchestrator"
    assert msg.message_type == "finding"
    assert set(msg.content.keys()) == {"symptom_patterns", "severity_score",
                                       "top_conditions", "confidence"}

    # Plan registered with the four intended actions and all four executed.
    plans = [p for p in _plans_in(store) if p.agent_id == "specialist_a"]
    assert len(plans) == 1
    assert plans[0].intended_actions == [
        "extract_symptoms", "lookup_conditions",
        "score_severity", "summarize_findings",
    ]
    assert plans[0].actual_actions == plans[0].intended_actions
    assert plans[0].deviations == []


# ---------------------------------------------------------------------------
# Specialist B
# ---------------------------------------------------------------------------

def test_specialist_b_writes_memory_and_sends_finding(store, loggers):
    """LLM emits candidates; textbook_search and guideline_lookup mocked."""
    fake_candidates = ["Acute myocardial infarction", "Pulmonary embolism"]
    llm = _fake_llm(fake_candidates)

    fake_docs = [
        {"doc_id": "Harrison__0001", "text": "ST-elevation MI management ...",
         "score": 0.91, "source_file": "InternalMed_Harrison.txt"},
        {"doc_id": "Harrison__0002", "text": "Reperfusion therapy and PCI ...",
         "score": 0.83, "source_file": "InternalMed_Harrison.txt"},
        {"doc_id": "Pathoma__0010",  "text": "Coronary artery thrombosis ...",
         "score": 0.77, "source_file": "Pathoma_Husain.txt"},
        {"doc_id": "FA__0050",       "text": "Aspirin in ACS ...",
         "score": 0.71, "source_file": "First_Aid_Step2.txt"},
        {"doc_id": "Robbins__0099",  "text": "Myocardial necrosis pathology ...",
         "score": 0.65, "source_file": "Pathology_Robbins.txt"},
    ]
    fake_textbook = MagicMock(return_value=fake_docs)

    def fake_guideline_lookup(condition: str) -> dict:
        if "myocardial" in condition.lower():
            return {
                "condition":  "Myocardial infarction",
                "summary":    "STUB",
                "source":     "synthetic/medqa-derived",
                "match":      "Myocardial infarction",
                "match_score": 0.92,
            }
        return {"match": None}

    agent = SpecialistB(
        agent_id="specialist_b",
        textbook_search_fn=fake_textbook,
        guideline_lookup_fn=fake_guideline_lookup,
        orchestrator_id="orchestrator",
        k_docs=5,
        top_evidence_n=3,
        llm=llm,
        **loggers,
    )

    out = agent.run({"patient_case": CASE_TEXT})

    assert set(out.keys()) == {"retrieved_docs", "top_evidence",
                               "guideline_matches", "retrieval_confidence"}

    mem = loggers["memory_logger"].for_agent("specialist_b")
    assert set(mem.keys()) == {"retrieved_docs", "top_evidence",
                               "guideline_matches", "retrieval_confidence"}
    assert mem["retrieved_docs"] == fake_docs
    assert len(mem["top_evidence"]) == 3
    assert mem["top_evidence"][0]["doc_id"] == "Harrison__0001"
    # mean of top-3 scores = (0.91 + 0.83 + 0.77) / 3
    assert mem["retrieval_confidence"] == pytest.approx(
        round((0.91 + 0.83 + 0.77) / 3, 4)
    )
    # Two candidates queried; first should match, second should miss.
    assert len(mem["guideline_matches"]) == 2
    assert mem["guideline_matches"][0]["queried"] == "Acute myocardial infarction"
    assert mem["guideline_matches"][0]["match"] == "Myocardial infarction"
    assert mem["guideline_matches"][1]["match"] is None

    # External calls. Three LLM invocations: generate_plan,
    # candidate-condition extraction, and the LLM-gated tool-selection
    # decision for textbook_search/guideline_lookup. The fake LLM returns
    # the same candidate list for every call, so the gating decision
    # parses as None and falls back to "run both with default queries".
    assert llm.invoke.call_count == 3
    fake_textbook.assert_called_once_with(CASE_TEXT, k=5)

    # Finding message to the orchestrator with the slim summary.
    msgs = [m for m in _messages_in(store) if m.sender == "specialist_b"]
    assert len(msgs) == 1
    msg = msgs[0]
    assert msg.receiver == "orchestrator"
    assert msg.message_type == "finding"
    assert set(msg.content.keys()) == {"n_docs", "top_evidence",
                                       "guideline_matches", "retrieval_confidence"}
    assert msg.content["n_docs"] == 5

    # Four-action plan, no deviations.
    plans = [p for p in _plans_in(store) if p.agent_id == "specialist_b"]
    assert len(plans) == 1
    assert plans[0].intended_actions == [
        "extract_candidate_conditions", "textbook_search",
        "guideline_lookup", "summarize_findings",
    ]
    assert plans[0].actual_actions == plans[0].intended_actions
    assert plans[0].deviations == []


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

def test_synthesizer_reads_memories_and_returns_structured_diagnosis(store, loggers):
    """LLM returns strict JSON; assert parse + memory write of final_output."""
    # Pre-populate both specialists' memory so the synthesizer has something to read.
    mem_logger: MemoryLogger = loggers["memory_logger"]
    mem_a = mem_logger.for_agent("specialist_a")
    mem_a["symptom_patterns"] = ["chest pain", "dyspnea"]
    mem_a["severity_score"]   = 0.85
    mem_a["top_conditions"]   = [["Acute MI", 0.9], ["PE", 0.4]]
    mem_a["confidence"]       = 0.8

    mem_b = mem_logger.for_agent("specialist_b")
    mem_b["retrieved_docs"]       = []
    mem_b["top_evidence"]         = [{"doc_id": "x", "score": 0.9,
                                      "source_file": "InternalMed_Harrison.txt",
                                      "snippet": "STEMI..."}]
    mem_b["guideline_matches"]    = [{"queried": "Acute MI", "match": "MI",
                                      "match_score": 0.95}]
    mem_b["retrieval_confidence"] = 0.9

    expected_payload = {
        "final_diagnosis": "Acute ST-elevation myocardial infarction",
        "confidence":      0.92,
        "differential":    ["Pulmonary embolism", "Aortic dissection"],
        "rationale":       "Classic STEMI presentation; specialists agree.",
    }
    # Wrap the JSON in some preamble to ensure the defensive parser strips it.
    raw_response = "Sure! Here is the diagnosis:\n\n" + json.dumps(expected_payload)
    llm = _fake_llm(raw_response)

    agent = Synthesizer(
        agent_id="synthesizer",
        specialist_a_id="specialist_a",
        specialist_b_id="specialist_b",
        llm=llm,
        **loggers,
    )

    out = agent.run({
        "patient_case": CASE_TEXT,
        "options": {"A": "PE", "B": "Acute ST-elevation myocardial infarction",
                    "C": "Aortic dissection", "D": "Pneumothorax"},
    })

    # Return contract — the original four keys are still present (backward
    # compat) plus the new option-level fields, all defaulting empty when
    # the LLM doesn't supply them.
    assert {"final_diagnosis", "confidence", "differential", "rationale",
            "predicted_letter", "predicted_text",
            "option_analysis", "supporting_evidence_ids"} <= set(out.keys())
    assert out["final_diagnosis"] == expected_payload["final_diagnosis"]
    assert out["confidence"] == pytest.approx(0.92)
    assert out["differential"] == expected_payload["differential"]
    assert out["rationale"] == expected_payload["rationale"]
    # Old-shape response → predicted_text mirrors final_diagnosis,
    # predicted_letter / option_analysis / supporting_evidence_ids stay empty.
    assert out["predicted_text"] == expected_payload["final_diagnosis"]
    assert out["predicted_letter"] == ""
    assert out["option_analysis"] == []
    assert out["supporting_evidence_ids"] == []

    # The result was written into the synthesizer's own memory.
    syn_mem = mem_logger.for_agent("synthesizer")
    assert "final_output" in syn_mem
    assert syn_mem["final_output"] == out

    # The synthesizer makes no tool calls; it makes exactly one LLM call.
    assert llm.invoke.call_count == 1
    prompt = llm.invoke.call_args.args[0]
    # Prompt embeds both memory dumps so the LLM has something to synthesise from.
    assert "symptom_patterns" in prompt
    assert "guideline_matches" in prompt

    # Plan: two-action, no deviations.
    plans = [p for p in _plans_in(store) if p.agent_id == "synthesizer"]
    assert len(plans) == 1
    assert plans[0].intended_actions == ["read_specialist_memories", "synthesize_diagnosis"]
    assert plans[0].actual_actions   == plans[0].intended_actions
    assert plans[0].deviations == []


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

# A reusable trio of stand-ins used by every orchestrator test below.
class _FakeSpecialist:
    """Minimal specialist that records its calls and emits a finding message."""

    def __init__(
        self,
        agent_id: str,
        finding: Dict[str, Any],
        message_logger: Any,
        call_log: List[Dict[str, Any]],
    ) -> None:
        self.agent_id = agent_id
        self._finding = finding
        self._message_logger = message_logger
        self._call_log = call_log
        self.payloads: List[Dict[str, Any]] = []

    def run(self, payload: dict) -> dict:
        self.payloads.append(dict(payload))
        self._call_log.append({"agent_id": self.agent_id, "payload": dict(payload)})
        self._message_logger.send(
            sender=self.agent_id,
            receiver="orchestrator",
            message_type="finding",
            content=self._finding,
        )
        return self._finding


class _FakeSynthesizer:
    agent_id = "synthesizer"

    def __init__(self, output: Dict[str, Any], call_log: List[Dict[str, Any]]) -> None:
        self._output = output
        self._call_log = call_log
        self.last_payload: Dict[str, Any] = {}

    def run(self, payload: dict) -> dict:
        self._call_log.append({"agent_id": self.agent_id, "payload": dict(payload)})
        self.last_payload = payload
        return self._output


def _orchestrator_actions(store: TrajectoryStore) -> List[str]:
    return [
        e.action for e in store.get_full_record(TASK_ID).xai_data.trajectory
        if e.agent_id == "orchestrator" and e.event_type == "action"
    ]


def test_orchestrator_routes_to_specialists_and_synthesizer(store, loggers):
    """No LLM wired → fall back to the deterministic A → B → synth rotation."""
    call_log: List[Dict[str, Any]] = []
    spec_a = _FakeSpecialist(
        "specialist_a", {"top_conditions": [("Acute MI", 1.0)]},
        loggers["message_logger"], call_log,
    )
    spec_b = _FakeSpecialist(
        "specialist_b", {"retrieval_confidence": 0.9},
        loggers["message_logger"], call_log,
    )
    synth_output = {
        "final_diagnosis": "Acute MI",
        "confidence":      0.93,
        "differential":    ["PE"],
        "rationale":       "Specialists converged on Acute MI.",
    }
    synth = _FakeSynthesizer(synth_output, call_log)

    agent = Orchestrator(
        agent_id="orchestrator",
        specialist_a=spec_a,
        specialist_b=spec_b,
        synthesizer=synth,
        llm=None,
        **loggers,
    )

    result = agent.run({
        "patient_case": CASE_TEXT,
        "options": {"A": "PE", "B": "Acute MI"},
    })

    # Sequential ordering — A before B before synthesizer.
    assert [c["agent_id"] for c in call_log] == [
        "specialist_a", "specialist_b", "synthesizer",
    ]

    # Top-level return wires through the synthesizer's output.
    assert result["final_output"] == synth_output
    assert result["specialist_a_id"] == "specialist_a"
    assert result["specialist_b_id"] == "specialist_b"

    # The handoff payload to the synthesizer carried both specialist agent_ids.
    assert synth.last_payload["specialist_a_id"] == "specialist_a"
    assert synth.last_payload["specialist_b_id"] == "specialist_b"

    # Plan registers the available action set; with llm=None generate_plan
    # returns the full list. The dynamic loop interleaves routing_decision
    # events with the specialist calls, so the actual sequence is longer
    # than the available-actions list — but every action is allowed.
    plans = [p for p in _plans_in(store) if p.agent_id == "orchestrator"]
    assert len(plans) == 1
    assert plans[0].intended_actions == _ORCHESTRATOR_PLAN
    assert _orchestrator_actions(store) == [
        "decompose_case",
        "routing_decision",
        "route_to_specialist_a",
        "routing_decision",
        "route_to_specialist_b",
        "routing_decision",
        "handoff_to_synthesizer",
    ]
    # Every executed action is one of the registered intended actions, so
    # the symmetric-diff deviation list is empty.
    assert plans[0].deviations == []

    # Orchestrator collected both specialists' finding messages via the
    # message channel (not just via return values).
    findings = agent.collected_findings()
    senders = sorted(f["sender"] for f in findings)
    assert senders == ["specialist_a", "specialist_b"]
    for f in findings:
        assert f["type"] == "finding"


# ---------------------------------------------------------------------------
# Dynamic LLM-driven routing
# ---------------------------------------------------------------------------

def _routing_llm(plan: List[str], *decisions: Dict[str, Any]) -> MagicMock:
    """
    Build a fake LLM that recognises the orchestrator's two prompt families:

      * generate_plan ("Available actions:" substring)         → returns ``plan``
      * routing decisions ("Choose the next action" substring) → returns the
        next entry from ``decisions``, in order. After the list runs out the
        LLM keeps replaying the last decision so tests for the max-iterations
        guard can drive an infinite "always re-call A" stream from a single
        decision payload.
    """
    fake = MagicMock()
    decision_iter = iter(decisions)
    last_decision: Dict[str, Any] = decisions[-1] if decisions else {
        "next_action": "synthesize", "reason": "default", "feedback_to_specialist": "",
    }

    def router(prompt: str) -> MagicMock:
        nonlocal last_decision
        if "Available actions" in prompt:
            return _resp(plan)
        if "Choose the next action" in prompt:
            try:
                payload = next(decision_iter)
            except StopIteration:
                payload = last_decision
            else:
                last_decision = payload
            return _resp(payload)
        return _resp([])

    fake.invoke.side_effect = router
    return fake


def _resp(content: Any) -> MagicMock:
    r = MagicMock()
    r.content = content if isinstance(content, str) else json.dumps(content)
    return r


def test_orchestrator_iteration_0_synthesize_is_overridden_by_guard(store, loggers):
    """
    LLM votes 'synthesize' on iteration 0 with no specialists called yet →
    the orchestrator's iteration-0 guard MUST override that to a specialist
    call. Synthesizing with zero findings is wrong by definition; specialists
    are how findings get gathered.

    After the guard fires, Specialist A runs once. The fake LLM then replays
    its last decision ('synthesize'), which is now valid (A has produced
    findings), and the orchestrator hands off.
    """
    call_log: List[Dict[str, Any]] = []
    spec_a = _FakeSpecialist(
        "specialist_a", {"top_conditions": [("Acute MI", 0.7)]},
        loggers["message_logger"], call_log,
    )
    spec_b = _FakeSpecialist("specialist_b", {}, loggers["message_logger"], call_log)
    synth = _FakeSynthesizer(
        {"final_diagnosis": "Acute MI", "confidence": 0.7,
         "differential": [], "rationale": "After A's findings."},
        call_log,
    )

    llm = _routing_llm(
        _ORCHESTRATOR_PLAN,
        {"next_action": "synthesize",
         "reason": "case is simple; existing context suffices",
         "feedback_to_specialist": ""},
    )

    agent = Orchestrator(
        agent_id="orchestrator",
        specialist_a=spec_a, specialist_b=spec_b, synthesizer=synth,
        llm=llm, max_iterations=5,
        **loggers,
    )
    agent.run({"patient_case": CASE_TEXT, "options": {"A": "PE", "B": "Acute MI"}})

    # Guard fired: specialist_a ran before the synthesizer; B never called.
    assert [c["agent_id"] for c in call_log] == ["specialist_a", "synthesizer"]

    # Trajectory: decompose, guarded routing_decision, route_to_specialist_a,
    # second routing_decision (now allowed to synthesize), handoff.
    assert _orchestrator_actions(store) == [
        "decompose_case",
        "routing_decision",
        "route_to_specialist_a",
        "routing_decision",
        "handoff_to_synthesizer",
    ]

    # The first routing_decision records the guard override in its outcome.
    routing_events = [
        e for e in store.get_full_record(TASK_ID).xai_data.trajectory
        if e.agent_id == "orchestrator" and e.action == "routing_decision"
    ]
    assert len(routing_events) == 2
    assert "iteration 0 guard" in routing_events[0].outcome
    assert routing_events[0].action_inputs["next_action"] == "call_specialist_a"


def test_orchestrator_iterative_recalls_specialist_with_feedback(store, loggers):
    """
    LLM re-calls Specialist A once with feedback, then synthesizes.

    The feedback string must be threaded into the second specialist run's
    input payload under ``feedback_from_orchestrator`` so the specialist's
    prompt can react to it.
    """
    call_log: List[Dict[str, Any]] = []
    spec_a = _FakeSpecialist(
        "specialist_a", {"top_conditions": [("MI", 0.8)]},
        loggers["message_logger"], call_log,
    )
    spec_b = _FakeSpecialist(
        "specialist_b", {}, loggers["message_logger"], call_log,
    )
    synth = _FakeSynthesizer(
        {"final_diagnosis": "Acute MI", "confidence": 0.85,
         "differential": [], "rationale": "After follow-up from A."},
        call_log,
    )

    llm = _routing_llm(
        _ORCHESTRATOR_PLAN,
        {"next_action": "call_specialist_a",
         "reason": "need symptom analysis first",
         "feedback_to_specialist": ""},
        {"next_action": "call_specialist_a",
         "reason": "low confidence — re-extract symptoms",
         "feedback_to_specialist": "focus on ECG findings"},
        {"next_action": "synthesize",
         "reason": "second pass resolved the ambiguity",
         "feedback_to_specialist": ""},
    )

    agent = Orchestrator(
        agent_id="orchestrator",
        specialist_a=spec_a, specialist_b=spec_b, synthesizer=synth,
        llm=llm, max_iterations=5,
        **loggers,
    )
    agent.run({"patient_case": CASE_TEXT, "options": {}})

    # Two A calls, no B, then synth.
    assert [c["agent_id"] for c in call_log] == [
        "specialist_a", "specialist_a", "synthesizer",
    ]

    # First A call has no feedback; second A call carries the LLM's feedback string.
    assert "feedback_from_orchestrator" not in spec_a.payloads[0]
    assert spec_a.payloads[1]["feedback_from_orchestrator"] == "focus on ECG findings"

    # Trajectory has three routing_decision events sandwiched between
    # the two specialist calls and the final handoff.
    assert _orchestrator_actions(store) == [
        "decompose_case",
        "routing_decision",
        "route_to_specialist_a",
        "routing_decision",
        "route_to_specialist_a",
        "routing_decision",
        "handoff_to_synthesizer",
    ]

    # The middle routing event records the feedback string in its inputs
    # (not just the action), so the dashboard can show what was passed back.
    middle = [
        e for e in store.get_full_record(TASK_ID).xai_data.trajectory
        if e.agent_id == "orchestrator" and e.action == "routing_decision"
    ][1]
    assert middle.action_inputs["feedback_to_specialist"] == "focus on ECG findings"


def test_orchestrator_max_iterations_guard_forces_synthesis(store, loggers):
    """
    LLM keeps voting 'call_specialist_a' forever → after max_iterations the
    orchestrator forces the handoff to the synthesizer rather than looping.
    """
    call_log: List[Dict[str, Any]] = []
    spec_a = _FakeSpecialist(
        "specialist_a", {"top_conditions": []},
        loggers["message_logger"], call_log,
    )
    spec_b = _FakeSpecialist(
        "specialist_b", {}, loggers["message_logger"], call_log,
    )
    synth = _FakeSynthesizer(
        {"final_diagnosis": "Inconclusive", "confidence": 0.3,
         "differential": [], "rationale": "Forced after max iterations."},
        call_log,
    )

    # Only one decision payload — the helper replays it indefinitely.
    llm = _routing_llm(
        _ORCHESTRATOR_PLAN,
        {"next_action": "call_specialist_a",
         "reason": "still want more from A",
         "feedback_to_specialist": ""},
    )

    max_iters = 3
    agent = Orchestrator(
        agent_id="orchestrator",
        specialist_a=spec_a, specialist_b=spec_b, synthesizer=synth,
        llm=llm, max_iterations=max_iters,
        **loggers,
    )
    agent.run({"patient_case": CASE_TEXT, "options": {}})

    # Specialist A is called exactly `max_iters` times; B is never called;
    # synthesizer is called once at the forced handoff.
    a_calls = [c for c in call_log if c["agent_id"] == "specialist_a"]
    b_calls = [c for c in call_log if c["agent_id"] == "specialist_b"]
    synth_calls = [c for c in call_log if c["agent_id"] == "synthesizer"]
    assert len(a_calls) == max_iters
    assert len(b_calls) == 0
    assert len(synth_calls) == 1

    # The trajectory contains exactly `max_iters + 1` routing_decision events
    # (max_iters real ones inside the loop, one synthetic "forced" entry on
    # exit) and the final handoff is annotated as forced.
    actions = _orchestrator_actions(store)
    assert actions[0] == "decompose_case"
    assert actions[-1] == "handoff_to_synthesizer"
    assert actions.count("routing_decision") == max_iters + 1
    assert actions.count("route_to_specialist_a") == max_iters

    # The last routing_decision before handoff has the forced-synthesis outcome.
    routing_events = [
        e for e in store.get_full_record(TASK_ID).xai_data.trajectory
        if e.agent_id == "orchestrator" and e.action == "routing_decision"
    ]
    forced = routing_events[-1]
    assert forced.action_inputs.get("forced") is True
    assert "max_iterations" in forced.outcome.lower()

    # The handoff event's outcome is suffixed with "(forced)" so a dashboard
    # consumer can flag the run as truncated.
    handoff = [
        e for e in store.get_full_record(TASK_ID).xai_data.trajectory
        if e.agent_id == "orchestrator" and e.action == "handoff_to_synthesizer"
    ][0]
    assert "(forced)" in handoff.outcome


# ---------------------------------------------------------------------------
# Memory-write traceability (regression for "No triggering event linked")
# ---------------------------------------------------------------------------

def _writes_for(store: TrajectoryStore, agent_id: str):
    """All MemoryDiff write rows owned by `agent_id` for the test task."""
    return [
        d for d in store.get_full_record(TASK_ID).xai_data.memory_diffs
        if d.agent_id == agent_id and d.operation == "write"
    ]


def _event_by_id(store: TrajectoryStore, event_id: str):
    for e in store.get_full_record(TASK_ID).xai_data.trajectory:
        if e.event_id == event_id:
            return e
    return None


class TestMemoryWriteTraceability:
    """
    Every MemoryDiff produced inside an agent's traced action must carry a
    `triggered_by_event_id` pointing at the enclosing TrajectoryEvent.

    Before the fix: the writes in `summarize_findings` happened *before*
    `log_action(...)` was called, so the contextvar was empty and every
    diff landed with `triggered_by_event_id=""`. The fix wraps the writes
    in `traced_action(...)` so the event is created first and the
    contextvar is set for the lifetime of the block.
    """

    def test_specialist_a_writes_link_to_summarize_findings_event(self, store, loggers):
        agent = SpecialistA(
            agent_id="specialist_a",
            symptom_lookup_fn=lambda s: {"related_conditions": [("MI", 0.5)]},
            severity_scorer_fn=lambda syms: 0.5,
            orchestrator_id="orchestrator",
            llm=_fake_llm(["chest pain"]),
            **loggers,
        )
        agent.run({"patient_case": CASE_TEXT})

        writes = _writes_for(store, "specialist_a")
        keys = {w.key for w in writes}
        # All four canonical keys must be present...
        assert keys == {
            "symptom_patterns", "severity_score",
            "top_conditions", "confidence",
        }
        # ...and every one must point at a real summarize_findings event.
        for w in writes:
            assert w.triggered_by_event_id, (
                f"specialist_a write {w.key!r} is unlinked "
                f"(triggered_by_event_id is empty)"
            )
            event = _event_by_id(store, w.triggered_by_event_id)
            assert event is not None, (
                f"specialist_a write {w.key!r} points at "
                f"{w.triggered_by_event_id!r} which does not exist in trajectory"
            )
            assert event.action == "summarize_findings"
            assert event.agent_id == "specialist_a"

        # All four writes share the same event_id (one summarize_findings event).
        assert len({w.triggered_by_event_id for w in writes}) == 1

    def test_specialist_b_writes_link_to_summarize_findings_event(self, store, loggers):
        fake_docs = [
            {"doc_id": "d1", "text": "evidence one", "score": 0.8,
             "source_file": "x.txt"},
        ]
        agent = SpecialistB(
            agent_id="specialist_b",
            textbook_search_fn=lambda case, k: fake_docs,
            guideline_lookup_fn=lambda c: {"match": "MI"},
            orchestrator_id="orchestrator",
            llm=_fake_llm(["Acute MI"]),
            **loggers,
        )
        agent.run({"patient_case": CASE_TEXT})

        writes = _writes_for(store, "specialist_b")
        keys = {w.key for w in writes}
        assert keys == {
            "retrieved_docs", "top_evidence",
            "guideline_matches", "retrieval_confidence",
        }
        for w in writes:
            assert w.triggered_by_event_id, (
                f"specialist_b write {w.key!r} is unlinked"
            )
            event = _event_by_id(store, w.triggered_by_event_id)
            assert event is not None
            assert event.action == "summarize_findings"
            assert event.agent_id == "specialist_b"

        assert len({w.triggered_by_event_id for w in writes}) == 1

    def test_synthesizer_final_output_links_to_synthesize_diagnosis_event(self, store, loggers):
        # Pre-seed specialist memory so the Synthesizer has something to read.
        # Use _emit-bypassing dict access (super().__setitem__) — but the
        # production path here does want LoggedMemory to log seeded values
        # too, so just use the public API. Those seed-writes WILL link to
        # whatever happens to be in current_event_id at the time, which is
        # fine for this test (we only assert the synthesizer's own write).
        mem_logger: MemoryLogger = loggers["memory_logger"]
        mem_logger.for_agent("specialist_a")["symptom_patterns"] = ["x"]
        mem_logger.for_agent("specialist_b")["top_evidence"] = []

        agent = Synthesizer(
            agent_id="synthesizer",
            specialist_a_id="specialist_a",
            specialist_b_id="specialist_b",
            llm=_fake_llm({
                "final_diagnosis": "MI",
                "confidence": 0.9,
                "differential": [],
                "rationale": "MI fits.",
            }),
            **loggers,
        )
        agent.run({"patient_case": CASE_TEXT, "options": {}})

        writes = _writes_for(store, "synthesizer")
        # Synthesizer writes exactly one key during run() — final_output.
        final_writes = [w for w in writes if w.key == "final_output"]
        assert len(final_writes) == 1
        w = final_writes[0]
        assert w.triggered_by_event_id, "final_output write is unlinked"
        event = _event_by_id(store, w.triggered_by_event_id)
        assert event is not None
        assert event.action == "synthesize_diagnosis"
        assert event.agent_id == "synthesizer"

    def test_traced_action_resets_contextvar_on_exit(self, store, loggers):
        """
        After a traced_action block exits, the contextvar must be reset so
        any subsequent untraced write lands "outside_traced_action" rather
        than spuriously linking to the just-finished event.
        """
        from agentxai.xai.memory_logger import current_event_id

        agent = SpecialistA(
            agent_id="specialist_a",
            symptom_lookup_fn=lambda s: {"related_conditions": []},
            severity_scorer_fn=lambda syms: 0.0,
            orchestrator_id="orchestrator",
            llm=_fake_llm([]),
            **loggers,
        )
        # Inside the run, log_action sets the contextvar to whatever was
        # most recent. After run() returns, run-scoped state is unwound.
        agent.run({"patient_case": ""})
        # Contextvar isn't reset by the agent leaving scope (it's a global),
        # but a write performed *now* — outside any active traced_action —
        # would attribute to whatever the last action set it to. We
        # explicitly clear it to simulate a fresh, untraced write context.
        current_event_id.set("")
        mem = loggers["memory_logger"].for_agent("specialist_a")
        mem["adhoc_key"] = "adhoc_value"
        all_writes = _writes_for(store, "specialist_a")
        adhoc = [w for w in all_writes if w.key == "adhoc_key"]
        assert len(adhoc) == 1
        # No event links — the dashboard renders this as
        # "outside_traced_action".
        assert adhoc[0].triggered_by_event_id == ""

    def test_no_diffs_carry_unknown_event_ids(self, store, loggers):
        """
        Sweep: across A, B, and Synth, every non-empty triggered_by_event_id
        on a write diff must resolve to a real TrajectoryEvent in the same
        task. Catches the "linked but stale id" failure mode separately
        from the "unlinked" one.
        """
        agent_a = SpecialistA(
            agent_id="specialist_a",
            symptom_lookup_fn=lambda s: {"related_conditions": [("MI", 0.5)]},
            severity_scorer_fn=lambda syms: 0.5,
            orchestrator_id="orchestrator",
            llm=_fake_llm(["chest pain"]),
            **loggers,
        )
        agent_b = SpecialistB(
            agent_id="specialist_b",
            textbook_search_fn=lambda case, k: [
                {"doc_id": "d1", "text": "ev", "score": 0.5, "source_file": "x.txt"},
            ],
            guideline_lookup_fn=lambda c: {"match": "MI"},
            orchestrator_id="orchestrator",
            llm=_fake_llm(["MI"]),
            **loggers,
        )
        agent_a.run({"patient_case": CASE_TEXT})
        agent_b.run({"patient_case": CASE_TEXT})

        record = store.get_full_record(TASK_ID)
        event_ids = {e.event_id for e in record.xai_data.trajectory}
        for d in record.xai_data.memory_diffs:
            if d.operation != "write":
                continue
            if not d.triggered_by_event_id:
                continue   # explicitly outside_traced_action — fine
            assert d.triggered_by_event_id in event_ids, (
                f"diff for {d.agent_id}/{d.key} references missing event "
                f"{d.triggered_by_event_id!r}"
            )


# ---------------------------------------------------------------------------
# Synthesizer option-level reasoning
# ---------------------------------------------------------------------------

# A representative HIV confirmatory-testing question. Option C is the
# correct modern answer per CDC algorithm; "Western blot" (option A) is
# the older test the LLM is sometimes anchored on. The fake LLM payload
# below exercises the case where the model picks C correctly AND
# produces option_analysis that explicitly downgrades the Western blot
# option — i.e., the rationale-vs-pick alignment the new prompt enforces.
_HIV_OPTIONS = {
    "A": "Western blot",
    "B": "Repeat HIV antibody screening test in 6 months",
    "C": "HIV-1/HIV-2 antibody differentiation immunoassay",
    "D": "p24 antigen",
    "E": "HIV RNA viral load",
}


class TestSynthesizerOptionLevelReasoning:
    """End-to-end coverage of the new option_analysis / predicted_letter contract."""

    def _hiv_payload(self) -> Dict[str, Any]:
        """Reference HIV LLM payload covering every required new field."""
        return {
            "predicted_letter": "C",
            "predicted_text":   _HIV_OPTIONS["C"],
            "final_diagnosis":  _HIV_OPTIONS["C"],
            "confidence":       0.91,
            "differential":     [_HIV_OPTIONS["A"], _HIV_OPTIONS["E"]],
            "rationale": (
                "Per the current CDC HIV testing algorithm a reactive screening "
                "immunoassay is followed by an HIV-1/HIV-2 antibody "
                "differentiation immunoassay (option C) — Western blot was "
                "phased out in 2014. Option C also distinguishes HIV-1 from "
                "HIV-2, which neither RNA viral load (option E) nor a repeat "
                "screen (option B) does."
            ),
            "option_analysis": [
                {"letter": "A", "text": _HIV_OPTIONS["A"], "verdict": "incorrect",
                 "reason": "Western blot is no longer the recommended confirmatory test."},
                {"letter": "B", "text": _HIV_OPTIONS["B"], "verdict": "incorrect",
                 "reason": "Delays diagnosis; not the next step after a reactive screen."},
                {"letter": "C", "text": _HIV_OPTIONS["C"], "verdict": "correct",
                 "reason": "Standard confirmatory step in the current CDC algorithm."},
                {"letter": "D", "text": _HIV_OPTIONS["D"], "verdict": "partial",
                 "reason": "Useful for acute HIV but not the confirmatory step."},
                {"letter": "E", "text": _HIV_OPTIONS["E"], "verdict": "incorrect",
                 "reason": "Reserved for indeterminate differentiation results."},
            ],
            "supporting_evidence_ids": ["Harrison__0341", "FA__step1_HIV"],
        }

    def test_full_payload_round_trips_through_synthesizer(self, store, loggers):
        """
        Fixed fake LLM response → all eight contract keys parsed and the
        full structured option_analysis is preserved verbatim through the
        Synthesizer's normalisation.
        """
        # Pre-seed both specialist memories so the read step has content.
        loggers["memory_logger"].for_agent("specialist_a")["top_conditions"] = []
        loggers["memory_logger"].for_agent("specialist_b")["top_evidence"] = []

        payload = self._hiv_payload()
        agent = Synthesizer(
            agent_id="synthesizer",
            specialist_a_id="specialist_a",
            specialist_b_id="specialist_b",
            llm=_fake_llm(payload),
            **loggers,
        )
        out = agent.run({"patient_case": "HIV stem", "options": _HIV_OPTIONS})

        assert out["predicted_letter"] == "C"
        assert out["predicted_text"] == _HIV_OPTIONS["C"]
        assert out["final_diagnosis"] == _HIV_OPTIONS["C"]
        assert out["confidence"] == pytest.approx(0.91)
        # Option analysis is structured, sorted by letter, with one entry per option.
        analysis = out["option_analysis"]
        assert [e["letter"] for e in analysis] == ["A", "B", "C", "D", "E"]
        assert {e["verdict"] for e in analysis} == {"correct", "incorrect", "partial"}
        # The specific HIV regression: Western blot must be flagged INCORRECT,
        # not casually mentioned as "the confirmatory test".
        wb = next(e for e in analysis if e["letter"] == "A")
        assert wb["verdict"] == "incorrect"
        assert "no longer" in wb["reason"].lower() or "phased" in wb["reason"].lower()
        # Supporting evidence ids preserved verbatim.
        assert out["supporting_evidence_ids"] == ["Harrison__0341", "FA__step1_HIV"]

        # Persisted on the synthesizer's own memory under final_output.
        assert loggers["memory_logger"].for_agent("synthesizer")["final_output"] == out

    def test_predicted_letter_backfills_from_option_analysis_when_missing(
        self, store, loggers,
    ):
        """
        If the LLM forgets to emit `predicted_letter` but DID mark one
        option as `correct` in option_analysis, the parser backfills.
        """
        loggers["memory_logger"].for_agent("specialist_a")["x"] = "y"
        loggers["memory_logger"].for_agent("specialist_b")["x"] = "y"

        payload = self._hiv_payload()
        del payload["predicted_letter"]
        del payload["predicted_text"]
        del payload["final_diagnosis"]

        agent = Synthesizer(
            agent_id="synthesizer",
            specialist_a_id="specialist_a",
            specialist_b_id="specialist_b",
            llm=_fake_llm(payload),
            **loggers,
        )
        out = agent.run({"patient_case": "HIV stem", "options": _HIV_OPTIONS})
        assert out["predicted_letter"] == "C"
        assert out["predicted_text"] == _HIV_OPTIONS["C"]
        assert out["final_diagnosis"] == _HIV_OPTIONS["C"]

    def test_malformed_option_analysis_entries_are_dropped_not_rejected(
        self, store, loggers,
    ):
        """A bad entry in option_analysis must not nuke the whole field."""
        loggers["memory_logger"].for_agent("specialist_a")["x"] = "y"
        loggers["memory_logger"].for_agent("specialist_b")["x"] = "y"

        payload = self._hiv_payload()
        # Inject garbage entries: not a dict, missing letter, bogus verdict,
        # duplicate letter (should keep the first occurrence).
        payload["option_analysis"] = [
            "not a dict",
            {"letter": "?", "verdict": "correct"},          # bad letter
            {"text": "no letter", "verdict": "correct"},    # missing letter
            {"letter": "A", "text": _HIV_OPTIONS["A"],
             "verdict": "correct", "reason": "ok"},
            {"letter": "A", "text": "duplicate", "verdict": "incorrect",
             "reason": "should be ignored"},
            {"letter": "B", "text": _HIV_OPTIONS["B"],
             "verdict": "weird-verdict", "reason": "blank verdict"},
        ]
        # Drop predicted_letter so the option_analysis backfill kicks in too.
        del payload["predicted_letter"]
        del payload["predicted_text"]
        del payload["final_diagnosis"]

        agent = Synthesizer(
            agent_id="synthesizer",
            specialist_a_id="specialist_a",
            specialist_b_id="specialist_b",
            llm=_fake_llm(payload),
            **loggers,
        )
        out = agent.run({"patient_case": "HIV", "options": _HIV_OPTIONS})

        analysis = out["option_analysis"]
        letters = [e["letter"] for e in analysis]
        assert letters == ["A", "B"], (
            f"only the two valid entries should survive; got {letters}"
        )
        # Duplicate "A" dropped — first occurrence wins.
        a = next(e for e in analysis if e["letter"] == "A")
        assert a["text"] == _HIV_OPTIONS["A"]
        # Bogus verdict normalised to empty string, not raised.
        b = next(e for e in analysis if e["letter"] == "B")
        assert b["verdict"] == ""
        # Backfill still found a "correct" entry → predicted_letter populated.
        assert out["predicted_letter"] == "A"

    def test_legacy_four_key_response_still_parses(self, store, loggers):
        """
        Backward-compat regression: an old-shape response (no predicted_*,
        no option_analysis, no supporting_evidence_ids) still produces a
        complete eight-key output with empty defaults.
        """
        loggers["memory_logger"].for_agent("specialist_a")["x"] = "y"
        loggers["memory_logger"].for_agent("specialist_b")["x"] = "y"

        legacy = {
            "final_diagnosis": "Acute MI",
            "confidence":      0.8,
            "differential":    ["PE"],
            "rationale":       "Old-shape rationale.",
        }
        agent = Synthesizer(
            agent_id="synthesizer",
            specialist_a_id="specialist_a",
            specialist_b_id="specialist_b",
            llm=_fake_llm(legacy),
            **loggers,
        )
        out = agent.run({"patient_case": "x", "options": {"A": "Acute MI", "B": "PE"}})

        assert out["final_diagnosis"] == "Acute MI"
        # predicted_text mirrors final_diagnosis on legacy responses.
        assert out["predicted_text"] == "Acute MI"
        # No structured letter pick when LLM didn't supply one.
        assert out["predicted_letter"] == ""
        assert out["option_analysis"] == []
        assert out["supporting_evidence_ids"] == []

    def test_prompt_includes_option_level_instructions(self, store, loggers):
        """
        The new prompt must explicitly instruct the LLM to (a) ground in
        the listed options, (b) downgrade outdated knowledge, and (c)
        produce option_analysis. Catches regressions where someone
        truncates the prompt back to the old four-key contract.
        """
        loggers["memory_logger"].for_agent("specialist_a")["x"] = "y"
        loggers["memory_logger"].for_agent("specialist_b")["x"] = "y"

        llm = _fake_llm({"predicted_letter": "A", "predicted_text": "Test"})
        agent = Synthesizer(
            agent_id="synthesizer",
            specialist_a_id="specialist_a",
            specialist_b_id="specialist_b",
            llm=llm,
            **loggers,
        )
        agent.run({"patient_case": "x", "options": {"A": "Test"}})

        prompt = llm.invoke.call_args.args[0]
        # Schema-level: every new key is named in the contract block.
        for key in (
            "predicted_letter",
            "predicted_text",
            "option_analysis",
            "supporting_evidence_ids",
        ):
            assert key in prompt, f"prompt missing required field {key!r}"
        # Behaviour-level: the option-grounding rule is explicit.
        prompt_low = prompt.lower()
        assert "do not recommend" in prompt_low or "not listed" in prompt_low
        assert "older guideline" in prompt_low or "defer to the listed" in prompt_low
        assert "verdict" in prompt_low


# ---------------------------------------------------------------------------
# LLM-driven tool selection (Pillar 3 case-by-case variation)
# ---------------------------------------------------------------------------

class TestLLMDrivenToolGating:
    """
    The specialists no longer call their tools in a fixed order with fixed
    inputs. Each tool call is gated by an LLM decision so Pillar 3 (Tool
    Provenance) varies by case.

    These tests assert the negative case: when the LLM returns ``run: false``
    for a tool, the tool callable is never invoked and no ToolUseEvent is
    persisted for it. The tools are wrapped with ``traced_tool`` against a
    real ``ToolProvenanceLogger`` so the absence of a ToolUseEvent is checked
    end-to-end against the store, not just the mock call count.
    """

    @staticmethod
    def _wrap(tool_logger, called_by: str, tool_name: str, mock):
        from agentxai.xai.tool_provenance import traced_tool
        return traced_tool(tool_logger, called_by=called_by, tool_name=tool_name)(mock)

    @staticmethod
    def _tool_names(store: TrajectoryStore) -> List[str]:
        return [tc.tool_name for tc in store.get_full_record(TASK_ID).xai_data.tool_calls]

    @staticmethod
    def _action_event(store: TrajectoryStore, agent_id: str, action: str):
        for e in store.get_full_record(TASK_ID).xai_data.trajectory:
            if e.agent_id == agent_id and e.action == action:
                return e
        return None

    # ------------------------------------------------------------------
    # Specialist A
    # ------------------------------------------------------------------

    def test_specialist_a_skips_severity_when_llm_says_no(self, store, loggers):
        """severity_scorer is never called and never logged when the LLM gates it off."""
        from agentxai.xai.tool_provenance import ToolProvenanceLogger

        tool_logger = ToolProvenanceLogger(store, TASK_ID)
        sym_mock = MagicMock(return_value={"related_conditions": [("MI", 0.5)]})
        sev_mock = MagicMock(return_value=0.99)
        sym_fn = self._wrap(tool_logger, "specialist_a", "symptom_lookup", sym_mock)
        sev_fn = self._wrap(tool_logger, "specialist_a", "severity_scorer", sev_mock)

        # LLM responses, in invocation order:
        #   1. generate_plan          → all four actions
        #   2. extract_symptoms       → one symptom phrase
        #   3. _select_lookup_subset  → echo the symptom (gate stays open)
        #   4. _decide_score_severity → run: false, with a reason
        llm = _fake_llm_seq(
            ["extract_symptoms", "lookup_conditions", "score_severity", "summarize_findings"],
            ["chest pain"],
            ["chest pain"],
            {"run": False, "reason": "non-acute presentation; severity ladder not informative"},
        )

        agent = SpecialistA(
            agent_id="specialist_a",
            symptom_lookup_fn=sym_fn,
            severity_scorer_fn=sev_fn,
            orchestrator_id="orchestrator",
            llm=llm,
            **loggers,
        )
        out = agent.run({"patient_case": CASE_TEXT})

        # The tool callable was never invoked; severity stays at 0.0.
        assert sev_mock.call_count == 0
        assert out["severity_score"] == 0.0

        # No ToolUseEvent persisted for severity_scorer. symptom_lookup ran
        # once (the lookup gate stayed open) so the store isn't empty —
        # we're verifying *selective* skipping, not blanket skipping.
        names = self._tool_names(store)
        assert "severity_scorer" not in names
        assert names.count("symptom_lookup") == 1

        # The trajectory event for score_severity still exists (the action
        # was decided, just not executed) and its outcome carries the LLM's
        # reasoning so the dashboard surfaces *why* the tool was skipped.
        evt = self._action_event(store, "specialist_a", "score_severity")
        assert evt is not None
        outcome_low = evt.outcome.lower()
        assert "skipped" in outcome_low
        assert "non-acute" in outcome_low

    def test_specialist_a_runs_severity_when_llm_says_yes(self, store, loggers):
        """Counterpart: ``run: true`` → severity_scorer fires and is logged."""
        from agentxai.xai.tool_provenance import ToolProvenanceLogger

        tool_logger = ToolProvenanceLogger(store, TASK_ID)
        sev_mock = MagicMock(return_value=0.42)
        sym_fn = self._wrap(
            tool_logger, "specialist_a", "symptom_lookup",
            MagicMock(return_value={"related_conditions": []}),
        )
        sev_fn = self._wrap(tool_logger, "specialist_a", "severity_scorer", sev_mock)

        llm = _fake_llm_seq(
            ["extract_symptoms", "lookup_conditions", "score_severity", "summarize_findings"],
            ["chest pain"],
            ["chest pain"],
            {"run": True, "reason": "acute red-flag features present"},
        )

        agent = SpecialistA(
            agent_id="specialist_a",
            symptom_lookup_fn=sym_fn,
            severity_scorer_fn=sev_fn,
            orchestrator_id="orchestrator",
            llm=llm,
            **loggers,
        )
        out = agent.run({"patient_case": CASE_TEXT})

        assert sev_mock.call_count == 1
        assert out["severity_score"] == pytest.approx(0.42)
        assert "severity_scorer" in self._tool_names(store)

    # ------------------------------------------------------------------
    # Specialist B
    # ------------------------------------------------------------------

    def test_specialist_b_skips_textbook_when_llm_says_no(self, store, loggers):
        """textbook_search is never called and never logged when gated off."""
        from agentxai.xai.tool_provenance import ToolProvenanceLogger

        tool_logger = ToolProvenanceLogger(store, TASK_ID)
        tb_mock = MagicMock(return_value=[
            {"doc_id": "d1", "text": "evidence", "score": 0.9, "source_file": "x.txt"},
        ])
        gl_mock = MagicMock(return_value={"match": "MI"})
        tb_fn = self._wrap(tool_logger, "specialist_b", "textbook_search", tb_mock)
        gl_fn = self._wrap(tool_logger, "specialist_b", "guideline_lookup", gl_mock)

        # LLM responses:
        #   1. generate_plan
        #   2. _extract_candidates
        #   3. _decide_evidence_tools — textbook off, guideline on
        llm = _fake_llm_seq(
            ["extract_candidate_conditions", "textbook_search",
             "guideline_lookup", "summarize_findings"],
            ["Acute MI"],
            {
                "textbook_search":  {"run": False, "reason": "case is rich in structured data; "
                                                              "free-text retrieval would add noise",
                                     "query": ""},
                "guideline_lookup": {"run": True,  "reason": "candidate list is concrete",
                                     "query": ""},
            },
        )

        agent = SpecialistB(
            agent_id="specialist_b",
            textbook_search_fn=tb_fn,
            guideline_lookup_fn=gl_fn,
            orchestrator_id="orchestrator",
            llm=llm,
            **loggers,
        )
        out = agent.run({"patient_case": CASE_TEXT})

        assert tb_mock.call_count == 0
        assert gl_mock.call_count == 1            # one candidate looped
        assert out["retrieved_docs"] == []
        assert out["top_evidence"] == []
        assert out["retrieval_confidence"] == 0.0

        names = self._tool_names(store)
        assert "textbook_search" not in names
        assert names.count("guideline_lookup") == 1

        evt = self._action_event(store, "specialist_b", "textbook_search")
        assert evt is not None
        outcome_low = evt.outcome.lower()
        assert "skipped" in outcome_low
        assert "noise" in outcome_low

    def test_specialist_b_skips_guideline_when_llm_says_no(self, store, loggers):
        """guideline_lookup is never called and never logged when gated off."""
        from agentxai.xai.tool_provenance import ToolProvenanceLogger

        tool_logger = ToolProvenanceLogger(store, TASK_ID)
        tb_mock = MagicMock(return_value=[
            {"doc_id": "d1", "text": "evidence", "score": 0.9, "source_file": "x.txt"},
        ])
        gl_mock = MagicMock(return_value={"match": "MI"})
        tb_fn = self._wrap(tool_logger, "specialist_b", "textbook_search", tb_mock)
        gl_fn = self._wrap(tool_logger, "specialist_b", "guideline_lookup", gl_mock)

        llm = _fake_llm_seq(
            ["extract_candidate_conditions", "textbook_search",
             "guideline_lookup", "summarize_findings"],
            ["Acute MI"],
            {
                "textbook_search":  {"run": True,  "reason": "free-text helpful here",
                                     "query": ""},
                "guideline_lookup": {"run": False, "reason": "no actionable guideline for "
                                                              "this presentation",
                                     "query": ""},
            },
        )

        agent = SpecialistB(
            agent_id="specialist_b",
            textbook_search_fn=tb_fn,
            guideline_lookup_fn=gl_fn,
            orchestrator_id="orchestrator",
            llm=llm,
            **loggers,
        )
        out = agent.run({"patient_case": CASE_TEXT})

        assert tb_mock.call_count == 1
        assert gl_mock.call_count == 0
        assert out["guideline_matches"] == []

        names = self._tool_names(store)
        assert "guideline_lookup" not in names
        assert names.count("textbook_search") == 1

        evt = self._action_event(store, "specialist_b", "guideline_lookup")
        assert evt is not None
        outcome_low = evt.outcome.lower()
        assert "skipped" in outcome_low
        assert "no actionable guideline" in outcome_low

    def test_specialist_b_skips_both_when_llm_says_no(self, store, loggers):
        """Neither retrieval tool is called when the LLM gates both off."""
        from agentxai.xai.tool_provenance import ToolProvenanceLogger

        tool_logger = ToolProvenanceLogger(store, TASK_ID)
        tb_mock = MagicMock(return_value=[])
        gl_mock = MagicMock(return_value={"match": None})
        tb_fn = self._wrap(tool_logger, "specialist_b", "textbook_search", tb_mock)
        gl_fn = self._wrap(tool_logger, "specialist_b", "guideline_lookup", gl_mock)

        llm = _fake_llm_seq(
            ["extract_candidate_conditions", "textbook_search",
             "guideline_lookup", "summarize_findings"],
            ["Acute MI"],
            {
                "textbook_search":  {"run": False, "reason": "skip a",  "query": ""},
                "guideline_lookup": {"run": False, "reason": "skip b",  "query": ""},
            },
        )

        agent = SpecialistB(
            agent_id="specialist_b",
            textbook_search_fn=tb_fn,
            guideline_lookup_fn=gl_fn,
            orchestrator_id="orchestrator",
            llm=llm,
            **loggers,
        )
        agent.run({"patient_case": CASE_TEXT})

        assert tb_mock.call_count == 0
        assert gl_mock.call_count == 0
        # No ToolUseEvent persisted at all when both gates close.
        assert self._tool_names(store) == []
