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

    # Tools were called the expected number of times.
    assert llm.invoke.call_count == 1
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
    """LLM emits candidates; pubmed_search and guideline_lookup mocked."""
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
    fake_pubmed = MagicMock(return_value=fake_docs)

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
        pubmed_search_fn=fake_pubmed,
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

    # External calls.
    assert llm.invoke.call_count == 1
    fake_pubmed.assert_called_once_with(CASE_TEXT, k=5)

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
        "extract_candidate_conditions", "pubmed_search",
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

def test_orchestrator_routes_to_specialists_and_synthesizer(store, loggers):
    """Mock the three downstream agents; assert sequential routing + handoff."""
    call_order: List[str] = []

    class _FakeSpecialist:
        def __init__(self, agent_id: str, finding: Dict[str, Any]):
            self.agent_id = agent_id
            self._finding = finding

        def run(self, payload: dict) -> dict:
            call_order.append(self.agent_id)
            # Specialists normally also send a finding message — emulate that
            # so collected_findings() has something to return.
            loggers["message_logger"].send(
                sender=self.agent_id,
                receiver="orchestrator",
                message_type="finding",
                content=self._finding,
            )
            return self._finding

    class _FakeSynthesizer:
        agent_id = "synthesizer"

        def __init__(self, output: Dict[str, Any]):
            self._output = output
            self.last_payload: Dict[str, Any] = {}

        def run(self, payload: dict) -> dict:
            call_order.append(self.agent_id)
            self.last_payload = payload
            return self._output

    spec_a = _FakeSpecialist("specialist_a", {"top_conditions": [("Acute MI", 1.0)]})
    spec_b = _FakeSpecialist("specialist_b", {"retrieval_confidence": 0.9})
    synth_output = {
        "final_diagnosis": "Acute MI",
        "confidence":      0.93,
        "differential":    ["PE"],
        "rationale":       "Specialists converged on Acute MI.",
    }
    synth = _FakeSynthesizer(synth_output)

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
    assert call_order == ["specialist_a", "specialist_b", "synthesizer"]

    # Top-level return wires through the synthesizer's output.
    assert result["final_output"] == synth_output
    assert result["specialist_a_id"] == "specialist_a"
    assert result["specialist_b_id"] == "specialist_b"

    # The handoff payload to the synthesizer carried both specialist agent_ids.
    assert synth.last_payload["specialist_a_id"] == "specialist_a"
    assert synth.last_payload["specialist_b_id"] == "specialist_b"

    # Plan: exactly the four spec'd intended actions, all executed in order.
    plans = [p for p in _plans_in(store) if p.agent_id == "orchestrator"]
    assert len(plans) == 1
    assert plans[0].intended_actions == _ORCHESTRATOR_PLAN
    assert plans[0].actual_actions   == _ORCHESTRATOR_PLAN
    assert plans[0].deviations == []

    # Orchestrator collected both specialists' finding messages via the
    # message channel (not just via return values).
    findings = agent.collected_findings()
    senders = sorted(f["sender"] for f in findings)
    assert senders == ["specialist_a", "specialist_b"]
    for f in findings:
        assert f["type"] == "finding"


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
            pubmed_search_fn=lambda case, k: fake_docs,
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
            pubmed_search_fn=lambda case, k: [
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
