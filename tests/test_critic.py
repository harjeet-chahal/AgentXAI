"""
Unit tests for the self-critique loop introduced in run_pipeline.py.

Coverage:

* The Critic agent itself produces the strict three-key contract under
  both "looks fine" and "needs work" responses, and its memory is
  written under a dedicated ``critic`` agent_id so the dashboard's
  memory tab surfaces it.
* ``Pipeline._run_self_critique`` orchestrates the loop correctly:
    (a) high-confidence critique → no revision
    (b) low-confidence critique  → exactly one orchestrator re-run
    (c) two-thumbs-down critique → revision cap prevents a second re-run
* The ``self_critique`` TrajectoryEvent is written under agent_id="critic".
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from agentxai.agents.critic import Critic
from agentxai.data.schemas import AgentXAIRecord
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.memory_logger import MemoryLogger
from agentxai.xai.message_logger import MessageLogger
from agentxai.xai.plan_tracker import PlanTracker
from agentxai.xai.trajectory_logger import TrajectoryLogger

from run_pipeline import Pipeline


TASK_ID = "CRITIC-TEST-001"


# ---------------------------------------------------------------------------
# Shared fixtures + helpers
# ---------------------------------------------------------------------------

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


def _fake_llm(content: Any) -> MagicMock:
    fake = MagicMock()
    response = MagicMock()
    response.content = content if isinstance(content, str) else json.dumps(content)
    fake.invoke.return_value = response
    return fake


_OPTIONS = {
    "A": "Acute MI",
    "B": "Pulmonary embolism",
    "C": "Aortic dissection",
}

_FINAL_OUT_GOOD = {
    "rationale":        "Classic STEMI presentation with ST elevations in the inferior leads.",
    "predicted_letter": "A",
    "predicted_text":   "Acute MI",
    "confidence":       0.9,
    "differential":     ["Pulmonary embolism"],
    "final_diagnosis":  "Acute MI",
}


# ---------------------------------------------------------------------------
# Critic agent (unit)
# ---------------------------------------------------------------------------

class TestCriticAgent:
    """The Critic in isolation: prompt → strict three-key contract."""

    def test_no_revision_when_llm_says_so(self, store, loggers):
        critique_payload = {
            "needs_revision":         False,
            "missing_considerations": [],
            "confidence_in_critique": 0.85,
        }
        critic = Critic(agent_id="critic", llm=_fake_llm(critique_payload), **loggers)
        out = critic.run({
            "rationale":        _FINAL_OUT_GOOD["rationale"],
            "predicted_letter": "A",
            "predicted_text":   "Acute MI",
            "options":          _OPTIONS,
        })

        assert out["needs_revision"] is False
        assert out["missing_considerations"] == []
        assert out["confidence_in_critique"] == pytest.approx(0.85)

        # Memory written under the dedicated "critic" agent_id so the
        # dashboard's memory tab can surface it.
        critic_mem = loggers["memory_logger"].for_agent("critic")
        assert "critique" in critic_mem
        assert "missing_considerations" in critic_mem
        assert critic_mem["critique"]["needs_revision"] is False

    def test_revision_flag_propagates_with_gap_list(self, store, loggers):
        critique_payload = {
            "needs_revision":         True,
            "missing_considerations": [
                "Aortic dissection not adequately ruled out",
                "Pericarditis ECG mimics not addressed",
            ],
            "confidence_in_critique": 0.7,
        }
        critic = Critic(agent_id="critic", llm=_fake_llm(critique_payload), **loggers)
        out = critic.run({
            "rationale":        "Brief rationale ignoring distractors.",
            "predicted_letter": "A",
            "predicted_text":   "Acute MI",
            "options":          _OPTIONS,
        })

        assert out["needs_revision"] is True
        assert "Aortic dissection" in out["missing_considerations"][0]
        assert len(out["missing_considerations"]) == 2

    def test_malformed_response_collapses_to_no_revision(self, store, loggers):
        # LLM returns garbage — Critic must fail-safe to "no revision needed"
        # rather than blocking the pipeline.
        critic = Critic(agent_id="critic", llm=_fake_llm("not even json"), **loggers)
        out = critic.run({
            "rationale": "x", "predicted_letter": "A",
            "predicted_text": "x", "options": _OPTIONS,
        })
        assert out["needs_revision"] is False
        assert out["missing_considerations"] == []
        assert out["confidence_in_critique"] == 0.0

    def test_llm_invocation_failure_collapses_to_empty_critique(self, store, loggers):
        # If the LLM call raises, we must fail-safe to "no revision needed"
        # rather than blocking the pipeline behind a critic exception.
        broken = MagicMock()
        broken.invoke.side_effect = RuntimeError("network down")
        critic = Critic(agent_id="critic", llm=broken, **loggers)
        out = critic.run({
            "rationale": "x", "predicted_letter": "A",
            "predicted_text": "x", "options": _OPTIONS,
        })
        assert out == {
            "needs_revision":         False,
            "missing_considerations": [],
            "confidence_in_critique": 0.0,
        }


# ---------------------------------------------------------------------------
# Pipeline._run_self_critique loop logic
# ---------------------------------------------------------------------------

class _RecordingOrchestrator:
    """Mock orchestrator that returns a configurable final_output per call."""

    agent_id = "orchestrator"

    def __init__(self, outputs: List[Dict[str, Any]]) -> None:
        self._outputs = list(outputs)
        self.calls: List[Dict[str, Any]] = []

    def run(self, payload: dict) -> dict:
        self.calls.append(dict(payload))
        idx = min(len(self.calls) - 1, len(self._outputs) - 1)
        return {
            "final_output":    self._outputs[idx],
            "specialist_a_id": "specialist_a",
            "specialist_b_id": "specialist_b",
        }


class _RecordingCritic:
    """Mock critic that yields a sequence of canned critiques per call."""

    agent_id = "critic"

    def __init__(self, critiques: List[Dict[str, Any]]) -> None:
        self._critiques = list(critiques)
        self.calls: List[Dict[str, Any]] = []

    def run(self, payload: dict) -> dict:
        self.calls.append(dict(payload))
        idx = min(len(self.calls) - 1, len(self._critiques) - 1)
        return dict(self._critiques[idx])


def _make_loop_pipeline() -> Pipeline:
    """Build a Pipeline without invoking the LLM factory."""
    return Pipeline(llm=MagicMock())


def _se_critique_events(store: TrajectoryStore) -> List:
    record = store.get_full_record(TASK_ID)
    return [
        e for e in record.xai_data.trajectory
        if e.event_type == "self_critique"
    ]


class TestSelfCritiqueLoop:
    """End-to-end loop: high-conf skip, low-conf revise, cap honored."""

    def _build(self, store, loggers, orch_outputs, critiques):
        pipeline = _make_loop_pipeline()
        orchestrator = _RecordingOrchestrator(orch_outputs)
        critic = _RecordingCritic(critiques)
        agents = {
            "orchestrator": orchestrator,
            "critic":       critic,
        }
        return pipeline, agents, orchestrator, critic

    def test_high_confidence_case_skips_revision(self, store, loggers):
        """needs_revision=False → no orchestrator re-run, revision_count=0."""
        first_final = dict(_FINAL_OUT_GOOD)
        pipeline, agents, orch, critic = self._build(
            store, loggers,
            orch_outputs=[first_final],
            critiques=[{
                "needs_revision":         False,
                "missing_considerations": [],
                "confidence_in_critique": 0.92,
            }],
        )

        agent_payload = {"patient_case": "x", "options": _OPTIONS}
        revision_count, final, letter, text = pipeline._run_self_critique(
            agents=agents, loggers=loggers,
            agent_payload=agent_payload,
            final=first_final,
            predicted_letter="A",
            predicted_text="Acute MI",
            options=_OPTIONS,
        )

        assert revision_count == 0
        assert final == first_final
        assert letter == "A"
        assert text == "Acute MI"
        # Critic ran once, orchestrator NEVER (the initial orchestrator
        # call happens *outside* _run_self_critique).
        assert len(critic.calls) == 1
        assert len(orch.calls) == 0

        # The self_critique TrajectoryEvent is logged under agent_id=critic
        # so the dashboard surfaces it as a distinct row.
        events = _se_critique_events(store)
        assert len(events) == 1
        assert events[0].agent_id == "critic"
        assert events[0].outcome == "no_revision_needed"

    def test_low_confidence_case_triggers_one_revision(self, store, loggers):
        """needs_revision=True → orchestrator re-called once with gaps in payload."""
        first_final = {
            "rationale":        "Brief — distractors not addressed.",
            "predicted_letter": "A",
            "predicted_text":   "Acute MI",
            "confidence":       0.4,
            "final_diagnosis":  "Acute MI",
            "differential":     [],
        }
        revised_final = {
            "rationale":        "Aortic dissection ruled out by widened mediastinum absence; PE less likely given ECG.",
            "predicted_letter": "B",
            "predicted_text":   "Pulmonary embolism",
            "confidence":       0.78,
            "final_diagnosis":  "Pulmonary embolism",
            "differential":     ["Acute MI"],
        }
        pipeline, agents, orch, critic = self._build(
            store, loggers,
            orch_outputs=[revised_final],
            critiques=[{
                "needs_revision":         True,
                "missing_considerations": [
                    "Aortic dissection not adequately ruled out",
                    "Distractors B and C ignored",
                ],
                "confidence_in_critique": 0.8,
            }],
        )

        agent_payload = {"patient_case": "x", "options": _OPTIONS}
        revision_count, final, letter, text = pipeline._run_self_critique(
            agents=agents, loggers=loggers,
            agent_payload=agent_payload,
            final=first_final,
            predicted_letter="A",
            predicted_text="Acute MI",
            options=_OPTIONS,
        )

        # Revision applied → revision_count=1 and we adopt the second output.
        assert revision_count == 1
        assert final == revised_final
        assert letter == "B"
        assert text == "Pulmonary embolism"

        # The orchestrator was called exactly once (the revision).
        assert len(orch.calls) == 1
        revised_payload = orch.calls[0]
        # The missing_considerations list was injected into the payload.
        assert revised_payload["missing_considerations"] == [
            "Aortic dissection not adequately ruled out",
            "Distractors B and C ignored",
        ]
        # Original payload fields preserved.
        assert revised_payload["patient_case"] == "x"
        assert revised_payload["options"] == _OPTIONS

        # Self-critique event recorded with the revision-needed outcome.
        events = _se_critique_events(store)
        assert len(events) == 1
        assert events[0].outcome == "revision_needed"
        # The state_after field carries the structured critique summary.
        assert events[0].state_after["needs_revision"] is True
        assert events[0].state_after["confidence_in_critique"] == pytest.approx(0.8)

    def test_revision_cap_prevents_infinite_loop(self, store, loggers):
        """
        Even when the critique sequence would say "needs_revision" forever,
        the pipeline calls the orchestrator at most ONCE for revision and
        never invokes the Critic a second time. Cap = 1.
        """
        first_final = dict(_FINAL_OUT_GOOD)
        revised_final = {
            **_FINAL_OUT_GOOD,
            "rationale":        "Revised but still allegedly weak.",
            "confidence":       0.55,
        }

        # The mock would happily emit "needs_revision=True" 10 times if asked.
        always_unhappy = [{
            "needs_revision":         True,
            "missing_considerations": ["something else"],
            "confidence_in_critique": 0.9,
        }] * 10

        pipeline, agents, orch, critic = self._build(
            store, loggers,
            orch_outputs=[revised_final, revised_final, revised_final],
            critiques=always_unhappy,
        )

        agent_payload = {"patient_case": "x", "options": _OPTIONS}
        revision_count, final, letter, text = pipeline._run_self_critique(
            agents=agents, loggers=loggers,
            agent_payload=agent_payload,
            final=first_final,
            predicted_letter="A",
            predicted_text="Acute MI",
            options=_OPTIONS,
        )

        # Cap honored: orchestrator called exactly once, critic called
        # exactly once (no second consultation), revision_count == 1.
        assert revision_count == 1
        assert len(orch.calls) == 1, (
            f"orchestrator must be re-called at most once; got {len(orch.calls)}"
        )
        assert len(critic.calls) == 1, (
            f"critic must NOT be consulted a second time; got {len(critic.calls)}"
        )

        # The accepted answer is the revision (we don't roll back).
        assert final == revised_final

        # Exactly one self_critique event was logged across the whole run.
        events = _se_critique_events(store)
        assert len(events) == 1
