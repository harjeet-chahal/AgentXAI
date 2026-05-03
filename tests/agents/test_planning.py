"""
Tests for LLM-generated planning (Pillar 2).

Each specialist now calls ``self.generate_plan(case, available_actions)`` to
let the LLM pick a subset and ordering of the hardcoded action list. These
tests verify that when a mocked LLM returns a partial plan, only the chosen
actions are executed and recorded — both in the trajectory log and on the
plan tracker.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from agentxai.agents.specialist_a import SpecialistA
from agentxai.agents.specialist_b import SpecialistB
from agentxai.data.schemas import AgentXAIRecord
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.memory_logger import MemoryLogger
from agentxai.xai.message_logger import MessageLogger
from agentxai.xai.plan_tracker import PlanTracker
from agentxai.xai.trajectory_logger import TrajectoryLogger


TASK_ID = "AGENT-PLAN-TEST-001"
CASE_TEXT = (
    "A 58-year-old man presents with crushing substernal chest pain radiating "
    "to the left arm, dyspnea, and diaphoresis."
)


@pytest.fixture()
def store() -> TrajectoryStore:
    s = TrajectoryStore(db_url="sqlite:///:memory:")
    s.save_task(AgentXAIRecord(task_id=TASK_ID, source="test"))
    return s


@pytest.fixture()
def loggers(store: TrajectoryStore) -> Dict[str, Any]:
    return {
        "trajectory_logger": TrajectoryLogger(store, TASK_ID),
        "plan_tracker":      PlanTracker(store, TASK_ID, llm=None),
        "memory_logger":     MemoryLogger(store, TASK_ID),
        "message_logger":    MessageLogger(store, TASK_ID),
    }


def _resp(content: Any) -> MagicMock:
    """LangChain-style chat response with a .content attribute."""
    r = MagicMock()
    r.content = content if isinstance(content, str) else json.dumps(content)
    return r


def _routing_llm(routes: Dict[str, Any]) -> MagicMock:
    """
    Build a fake LLM that returns different responses based on a substring
    match against the prompt. ``routes`` maps a substring → response payload.
    The first matching key wins; falls through to "" → ``[]``.
    """
    fake = MagicMock()

    def router(prompt: str) -> MagicMock:
        for needle, payload in routes.items():
            if needle in prompt:
                return _resp(payload)
        return _resp([])

    fake.invoke.side_effect = router
    return fake


def _plans_in(store: TrajectoryStore, agent_id: str):
    return [
        p for p in store.get_full_record(TASK_ID).xai_data.plans
        if p.agent_id == agent_id
    ]


def _actions_in(store: TrajectoryStore, agent_id: str) -> List[str]:
    return [
        e.action for e in store.get_full_record(TASK_ID).xai_data.trajectory
        if e.agent_id == agent_id and e.event_type == "action"
    ]


def _messages_from(store: TrajectoryStore, sender: str):
    return [
        m for m in store.get_full_record(TASK_ID).xai_data.messages
        if m.sender == sender
    ]


# ---------------------------------------------------------------------------
# SpecialistA — partial plan
# ---------------------------------------------------------------------------

def test_specialist_a_partial_plan_runs_only_chosen_actions(store, loggers):
    """LLM returns only [extract_symptoms, score_severity]; the other two
    actions must be skipped — no log_action, no plan record, no memory."""
    chosen = ["extract_symptoms", "score_severity"]
    llm = _routing_llm({
        "Available actions": chosen,            # generate_plan
        "clinical NLP extractor": ["chest pain", "dyspnea"],  # extract_symptoms prompt
    })

    fake_lookup = MagicMock(return_value={"related_conditions": [("MI", 0.9)]})
    fake_severity = MagicMock(return_value=0.42)

    agent = SpecialistA(
        agent_id="specialist_a",
        symptom_lookup_fn=fake_lookup,
        severity_scorer_fn=fake_severity,
        orchestrator_id="orchestrator",
        llm=llm,
        **loggers,
    )

    out = agent.run({"patient_case": CASE_TEXT})

    # summarize_findings was skipped → no findings written, returned empty.
    assert out == {}

    # Plan reflects exactly the chosen actions, in the LLM's chosen order.
    plans = _plans_in(store, "specialist_a")
    assert len(plans) == 1
    assert plans[0].intended_actions == chosen
    assert plans[0].actual_actions == chosen
    assert plans[0].deviations == []

    # Trajectory: only the two chosen actions were logged.
    assert _actions_in(store, "specialist_a") == chosen

    # symptom_lookup belongs to the skipped lookup_conditions step, so it
    # must not have been called even though symptoms were extracted.
    fake_lookup.assert_not_called()
    fake_severity.assert_called_once_with(["chest pain", "dyspnea"])

    # Memory unchanged — summarize_findings is the only writer and it ran.
    mem = loggers["memory_logger"].for_agent("specialist_a")
    assert dict(mem) == {}

    # No finding message emitted (only summarize_findings sends one).
    assert _messages_from(store, "specialist_a") == []


# ---------------------------------------------------------------------------
# SpecialistB — partial plan
# ---------------------------------------------------------------------------

def test_specialist_b_partial_plan_runs_only_chosen_actions(store, loggers):
    """LLM picks a 2-action plan for SpecialistB; the other two are skipped."""
    chosen = ["textbook_search", "summarize_findings"]
    llm = _routing_llm({
        "Available actions": chosen,           # generate_plan
        "clinical reasoner":  ["Acute MI"],    # extract_candidate_conditions
    })

    fake_docs = [
        {"doc_id": "d1", "text": "ev one", "score": 0.8, "source_file": "x.txt"},
    ]
    fake_textbook = MagicMock(return_value=fake_docs)
    fake_guideline = MagicMock(return_value={"match": "MI"})

    agent = SpecialistB(
        agent_id="specialist_b",
        textbook_search_fn=fake_textbook,
        guideline_lookup_fn=fake_guideline,
        orchestrator_id="orchestrator",
        llm=llm,
        **loggers,
    )

    out = agent.run({"patient_case": CASE_TEXT})

    # Plan reflects the LLM's choice exactly.
    plans = _plans_in(store, "specialist_b")
    assert len(plans) == 1
    assert plans[0].intended_actions == chosen
    assert plans[0].actual_actions == chosen
    assert plans[0].deviations == []

    # Trajectory contains only the chosen actions.
    assert _actions_in(store, "specialist_b") == chosen

    # extract_candidate_conditions was skipped, so guideline_lookup (which
    # would otherwise iterate over candidates) had no candidates AND was
    # itself skipped → guideline_lookup_fn never called.
    fake_guideline.assert_not_called()
    # The candidate-extraction LLM prompt must also have been skipped.
    invoked_prompts = [c.args[0] for c in llm.invoke.call_args_list]
    assert not any("clinical reasoner" in p for p in invoked_prompts)

    # textbook_search ran and summarize_findings persisted its output.
    fake_textbook.assert_called_once()
    assert out["retrieved_docs"] == fake_docs
    assert out["top_evidence"][0]["doc_id"] == "d1"
    # No candidates were extracted, so guideline_matches is empty.
    assert out["guideline_matches"] == []

    # Exactly one finding message went out (from summarize_findings).
    msgs = _messages_from(store, "specialist_b")
    assert len(msgs) == 1
    assert msgs[0].message_type == "finding"


# ---------------------------------------------------------------------------
# Fallback: invalid / empty LLM response → full hardcoded list
# ---------------------------------------------------------------------------

def test_specialist_a_empty_plan_falls_back_to_full_list(store, loggers):
    """generate_plan returns []; agent must fall back to the full action list."""
    llm = _routing_llm({
        "Available actions": [],                    # empty plan from LLM
        "clinical NLP extractor": ["chest pain"],   # symptom extraction
    })
    agent = SpecialistA(
        agent_id="specialist_a",
        symptom_lookup_fn=lambda s: {"related_conditions": []},
        severity_scorer_fn=lambda syms: 0.0,
        orchestrator_id="orchestrator",
        llm=llm,
        **loggers,
    )
    agent.run({"patient_case": CASE_TEXT})

    plans = _plans_in(store, "specialist_a")
    assert len(plans) == 1
    assert plans[0].intended_actions == [
        "extract_symptoms", "lookup_conditions",
        "score_severity", "summarize_findings",
    ]
    assert plans[0].actual_actions == plans[0].intended_actions


def test_specialist_a_invalid_action_names_fall_back_to_full_list(store, loggers):
    """LLM returns names that are not in available_actions → full fallback."""
    llm = _routing_llm({
        "Available actions": ["bogus", "also_bogus"],
        "clinical NLP extractor": ["chest pain"],
    })
    agent = SpecialistA(
        agent_id="specialist_a",
        symptom_lookup_fn=lambda s: {"related_conditions": []},
        severity_scorer_fn=lambda syms: 0.0,
        orchestrator_id="orchestrator",
        llm=llm,
        **loggers,
    )
    agent.run({"patient_case": CASE_TEXT})

    plans = _plans_in(store, "specialist_a")
    assert plans[0].intended_actions == [
        "extract_symptoms", "lookup_conditions",
        "score_severity", "summarize_findings",
    ]
