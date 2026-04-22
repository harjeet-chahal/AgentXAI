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

    # Return contract.
    assert set(out.keys()) == {"final_diagnosis", "confidence",
                               "differential", "rationale"}
    assert out["final_diagnosis"] == expected_payload["final_diagnosis"]
    assert out["confidence"] == pytest.approx(0.92)
    assert out["differential"] == expected_payload["differential"]
    assert out["rationale"] == expected_payload["rationale"]

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
