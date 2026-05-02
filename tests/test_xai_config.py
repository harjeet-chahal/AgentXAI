"""
Tests for `agentxai.xai.config.XAIScoringConfig` and the wiring through
the three components (CausalDAGBuilder, CounterfactualEngine,
AccountabilityReportGenerator).

Two-part contract:
  1. Defaults reproduce historical behavior — every module-level constant
     that used to be hard-coded equals the corresponding config field.
  2. Custom configs change scoring as expected — passing an instance
     with a knob flipped propagates through every component cleanly.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from agentxai.data.schemas import (
    AccountabilityReport,
    AgentXAIRecord,
    MemoryDiff,
    TrajectoryEvent,
)
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.accountability import (
    AccountabilityReportGenerator,
    _RESP_WEIGHTS,
    _ROOT_ACTED_MSG_BONUS,
    _ROOT_MEM_BONUS,
    _ROOT_TOOL_BONUS,
    _ROOT_UPSTREAM_DISCOUNT,
    _QUESTION_TYPE_PRIORS,
    _TIE_EPSILON,
    _combine_signals,
    _fallback_explanation,
    _prior_for,
    _tied_top_agents,
)
from agentxai.xai.causal_dag import CausalDAGBuilder, _TEMPORAL_WEIGHT
from agentxai.xai.config import (
    DEFAULT_CONFIG,
    XAIScoringConfig,
    _DEFAULT_QUESTION_TYPE_PRIORS,
    _DEFAULT_RESPONSIBILITY_WEIGHTS,
)
from agentxai.xai.counterfactual_engine import (
    CounterfactualEngine,
    _CONF_WEIGHT,
    _DX_WEIGHT,
    _outcome_delta,
)


# ---------------------------------------------------------------------------
# 1) Defaults — every historical constant matches the config field
# ---------------------------------------------------------------------------

class TestDefaultsMatchHistoricalConstants:
    """A regression: the move-to-config refactor must not silently change a value."""

    def test_counterfactual_weights(self):
        assert DEFAULT_CONFIG.cf_dx_weight == pytest.approx(0.6)
        assert DEFAULT_CONFIG.cf_conf_weight == pytest.approx(0.4)
        # Module aliases match.
        assert _DX_WEIGHT == pytest.approx(DEFAULT_CONFIG.cf_dx_weight)
        assert _CONF_WEIGHT == pytest.approx(DEFAULT_CONFIG.cf_conf_weight)

    def test_causal_dag_edge_weights(self):
        assert DEFAULT_CONFIG.temporal_edge_weight == pytest.approx(0.3)
        assert DEFAULT_CONFIG.message_acted_upon_weight == pytest.approx(1.0)
        assert DEFAULT_CONFIG.message_ignored_weight == pytest.approx(0.5)
        assert _TEMPORAL_WEIGHT == pytest.approx(DEFAULT_CONFIG.temporal_edge_weight)

    def test_root_cause_bonuses(self):
        assert DEFAULT_CONFIG.root_tool_bonus == pytest.approx(1.5)
        assert DEFAULT_CONFIG.root_acted_msg_bonus == pytest.approx(1.0)
        assert DEFAULT_CONFIG.root_mem_bonus == pytest.approx(0.5)
        assert DEFAULT_CONFIG.root_upstream_discount == pytest.approx(0.3)
        # Aliases.
        assert _ROOT_TOOL_BONUS == pytest.approx(DEFAULT_CONFIG.root_tool_bonus)
        assert _ROOT_ACTED_MSG_BONUS == pytest.approx(DEFAULT_CONFIG.root_acted_msg_bonus)
        assert _ROOT_MEM_BONUS == pytest.approx(DEFAULT_CONFIG.root_mem_bonus)
        assert _ROOT_UPSTREAM_DISCOUNT == pytest.approx(DEFAULT_CONFIG.root_upstream_discount)

    def test_tie_epsilon(self):
        assert DEFAULT_CONFIG.tie_epsilon == pytest.approx(0.05)
        assert _TIE_EPSILON == pytest.approx(DEFAULT_CONFIG.tie_epsilon)

    def test_responsibility_weights_default(self):
        # Module alias is a copy of the config dict.
        assert _RESP_WEIGHTS == DEFAULT_CONFIG.responsibility_weights
        # Module alias also matches the canonical defaults exposed by config.
        assert _RESP_WEIGHTS == _DEFAULT_RESPONSIBILITY_WEIGHTS
        # Sums to 1 so unnormalised composite stays in [0, 1].
        assert sum(_RESP_WEIGHTS.values()) == pytest.approx(1.0)

    def test_question_type_priors_default(self):
        assert _QUESTION_TYPE_PRIORS == DEFAULT_CONFIG.question_type_priors
        assert _QUESTION_TYPE_PRIORS == _DEFAULT_QUESTION_TYPE_PRIORS
        # Required keys in the taxonomy.
        for qt in (
            "diagnosis", "treatment", "screening_or_test", "pharmacology",
            "mechanism", "risk_factor", "anatomy", "prognosis", "unknown",
        ):
            assert qt in _QUESTION_TYPE_PRIORS

    def test_module_aliases_are_frozen_copies(self):
        # Mutating the module alias must NOT poison the default config.
        original = dict(DEFAULT_CONFIG.responsibility_weights)
        _RESP_WEIGHTS["counterfactual"] = 0.0   # malicious mutation
        try:
            assert DEFAULT_CONFIG.responsibility_weights == original
        finally:
            # Restore so subsequent tests see the canonical default.
            _RESP_WEIGHTS["counterfactual"] = original["counterfactual"]


# ---------------------------------------------------------------------------
# 2) Custom configs change scoring as expected
# ---------------------------------------------------------------------------

class TestOutcomeDeltaWithCustomConfig:
    """`_outcome_delta` accepts a config and respects its weights."""

    def test_default_matches_old_behavior(self):
        orig = {"final_diagnosis": "A", "confidence": 0.9}
        perturbed = {"final_diagnosis": "B", "confidence": 0.5}
        # 0.6 * 1 + 0.4 * 0.4 = 0.76
        assert _outcome_delta(orig, perturbed) == pytest.approx(0.76)

    def test_dx_weight_dominant(self):
        cfg = XAIScoringConfig(cf_dx_weight=1.0, cf_conf_weight=0.0)
        orig = {"final_diagnosis": "A", "confidence": 0.9}
        flipped = {"final_diagnosis": "B", "confidence": 0.9}
        # Pure dx flip → 1.0; conf delta ignored.
        assert _outcome_delta(orig, flipped, cfg) == pytest.approx(1.0)

    def test_conf_weight_dominant(self):
        cfg = XAIScoringConfig(cf_dx_weight=0.0, cf_conf_weight=1.0)
        orig = {"final_diagnosis": "A", "confidence": 1.0}
        unchanged_dx = {"final_diagnosis": "A", "confidence": 0.0}
        assert _outcome_delta(orig, unchanged_dx, cfg) == pytest.approx(1.0)

    def test_explicit_None_falls_back_to_default(self):
        orig = {"final_diagnosis": "A", "confidence": 0.9}
        perturbed = {"final_diagnosis": "B", "confidence": 0.5}
        assert _outcome_delta(orig, perturbed, None) == _outcome_delta(orig, perturbed)


class TestCausalDAGBuilderConfigPropagation:
    """`CausalDAGBuilder` reads temporal + message weights from its config."""

    def _build_minimal_task(self):
        s = TrajectoryStore(db_url="sqlite:///:memory:")
        task_id = "DAG-CFG-TASK"
        s.save_task(AgentXAIRecord(task_id=task_id, source="test"))
        # Two events from the same agent → temporal edge between them.
        events = [
            TrajectoryEvent(event_id="e1", agent_id="specialist_a",
                            event_type="agent_action", action="x",
                            timestamp=1.0),
            TrajectoryEvent(event_id="e2", agent_id="specialist_a",
                            event_type="agent_action", action="y",
                            timestamp=2.0),
        ]
        for e in events:
            s.save_event(task_id, e)
        return s, task_id

    def test_default_temporal_weight(self):
        s, task_id = self._build_minimal_task()
        graph = CausalDAGBuilder(s).build(task_id)
        weight = graph["e1"]["e2"]["weight"]
        assert weight == pytest.approx(0.3)

    def test_custom_temporal_weight(self):
        s, task_id = self._build_minimal_task()
        cfg = XAIScoringConfig(temporal_edge_weight=0.7)
        graph = CausalDAGBuilder(s, config=cfg).build(task_id)
        assert graph["e1"]["e2"]["weight"] == pytest.approx(0.7)

    def test_message_weights_respect_config(self):
        # Build a tiny task where SpecialistA → Synthesizer message is
        # NOT acted_upon. Default = 0.5; custom = 0.2.
        from agentxai.data.schemas import AgentMessage

        def _build():
            s = TrajectoryStore(db_url="sqlite:///:memory:")
            task_id = "MSG-CFG-TASK"
            s.save_task(AgentXAIRecord(task_id=task_id, source="test"))
            s.save_event(task_id, TrajectoryEvent(
                event_id="src", agent_id="specialist_a",
                event_type="agent_action", action="emit", timestamp=1.0,
            ))
            s.save_event(task_id, TrajectoryEvent(
                event_id="tgt", agent_id="synthesizer",
                event_type="agent_action", action="recv", timestamp=3.0,
            ))
            s.save_message(task_id, AgentMessage(
                message_id="m1", sender="specialist_a",
                receiver="synthesizer", timestamp=2.0,
                acted_upon=False,
            ))
            return s, task_id

        s, task_id = _build()
        # Default: 0.5.
        graph = CausalDAGBuilder(s).build(task_id)
        assert graph["src"]["tgt"]["weight"] == pytest.approx(0.5)

        s, task_id = _build()
        cfg = XAIScoringConfig(message_ignored_weight=0.2)
        graph = CausalDAGBuilder(s, config=cfg).build(task_id)
        assert graph["src"]["tgt"]["weight"] == pytest.approx(0.2)


class TestPriorForWithCustomConfig:
    def test_default_neutral_for_unknown(self):
        assert _prior_for("unknown", "specialist_a") == 1.0
        assert _prior_for("unknown", "specialist_b") == 1.0

    def test_default_screening_shrinks_a(self):
        assert _prior_for("screening_or_test", "specialist_a") == 0.5

    def test_custom_priors_override(self):
        custom = {"screening_or_test": {"specialist_a": 0.1, "specialist_b": 0.9}}
        assert _prior_for("screening_or_test", "specialist_a",
                          priors=custom) == 0.1
        # Unknown agents → 1.0 fall-back.
        assert _prior_for("screening_or_test", "ghost", priors=custom) == 1.0


class TestCombineSignalsWithCustomWeights:
    def test_default_weights_match_resp_weights(self):
        all_ones = {k: 1.0 for k in _RESP_WEIGHTS}
        # Sum of weights = 1.0 → all-ones combined = 1.0.
        assert _combine_signals(all_ones) == pytest.approx(1.0)

    def test_custom_weights_change_total(self):
        # Make tool_impact dominate.
        custom = {"counterfactual": 0.0, "tool_impact": 1.0,
                  "message_efficacy": 0.0, "memory_used": 0.0,
                  "usefulness": 0.0, "causal_centrality": 0.0}
        signals = {"tool_impact": 0.7, "counterfactual": 0.5}
        assert _combine_signals(signals, weights=custom) == pytest.approx(0.7)


class TestTiedTopAgentsWithCustomEpsilon:
    def test_default_epsilon_groups_close_agents(self):
        scores = {"a": 0.50, "b": 0.49, "c": 0.30}
        tied = _tied_top_agents(scores)
        assert tied == ["a", "b"]

    def test_custom_epsilon_tight_excludes_close_runner_up(self):
        # epsilon=0.001 → only agents within 0.001 of the top are tied.
        # "a" is at the top (gap 0 < 0.001 → tied with itself); "b" is
        # 0.01 away → excluded.
        scores = {"a": 0.50, "b": 0.49, "c": 0.30}
        tied = _tied_top_agents(scores, epsilon=0.001)
        assert tied == ["a"]

    def test_custom_epsilon_wide_includes_more(self):
        scores = {"a": 0.50, "b": 0.49, "c": 0.40}
        tied = _tied_top_agents(scores, epsilon=0.20)
        assert set(tied) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# End-to-end: AccountabilityReportGenerator honours its config
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = text


class _FakeLLM:
    def invoke(self, prompt: str):
        return _FakeResponse("ok")


class _MockPipeline:
    def __init__(self, output: Dict[str, Any]):
        self.output = output

    def resume_from(self, snapshot, overrides):
        return dict(self.output)


def _build_resp_scenario():
    """Two-specialist task with tied observable signals."""
    s = TrajectoryStore(db_url="sqlite:///:memory:")
    task_id = "CFG-RESP-TASK"
    s.save_task(AgentXAIRecord(
        task_id=task_id,
        source="test",
        input={"patient_case": "x", "options": {}, "question_type": "unknown"},
        ground_truth={},
        system_output={"final_diagnosis": "X", "confidence": 0.9,
                       "correct": True, "rationale": "MI"},
    ))
    for ag in ("specialist_a", "specialist_b"):
        s.save_event(task_id, TrajectoryEvent(
            agent_id=ag, event_type="agent_action",
            action="diagnose", timestamp=1.0,
        ))
    s.save_memory_diff(task_id, MemoryDiff(
        agent_id="specialist_a", operation="write",
        key="top_conditions", value_after=["MI"],
    ))
    s.save_memory_diff(task_id, MemoryDiff(
        agent_id="specialist_b", operation="write",
        key="top_evidence", value_after=["MI"],
    ))
    return s, task_id


class TestAccountabilityReportGeneratorConfig:
    def test_default_config_reproduces_baseline(self):
        # Without an explicit config, behaviour is identical to before
        # the refactor. Smoke test: a tied-signal scenario gives a 50/50
        # split when question_type is unknown.
        s, task_id = _build_resp_scenario()
        gen = AccountabilityReportGenerator(
            store=s, pipeline=_MockPipeline({"final_diagnosis": "Y",
                                              "confidence": 0.5}),
            llm=_FakeLLM(),
        )
        report = gen.generate(task_id)
        a = report.agent_responsibility_scores["specialist_a"]
        b = report.agent_responsibility_scores["specialist_b"]
        assert a == pytest.approx(b, abs=1e-6)

    def test_custom_question_type_priors_propagate(self):
        # Override the priors so screening_or_test is strongly skewed
        # toward A (instead of B), then attach the question_type to the task.
        s, task_id = _build_resp_scenario()
        # Re-tag the task to a screening question.
        rec = s.get_full_record(task_id)
        rec.input["question_type"] = "screening_or_test"
        s.save_task(rec)

        cfg = XAIScoringConfig(
            question_type_priors={
                "screening_or_test": {"specialist_a": 1.0, "specialist_b": 0.1},
            },
        )
        gen = AccountabilityReportGenerator(
            store=s, pipeline=_MockPipeline({"final_diagnosis": "Y",
                                              "confidence": 0.5}),
            llm=_FakeLLM(), config=cfg,
        )
        report = gen.generate(task_id)
        a = report.agent_responsibility_scores["specialist_a"]
        b = report.agent_responsibility_scores["specialist_b"]
        # B's prior is now 0.1 → A should clearly outrank B.
        assert a > 0.7
        assert b < 0.3

    def test_custom_responsibility_weights_propagate(self):
        # Make `tool_impact` the only contributing signal; everything
        # else zero-weight. Specialist B has no tool calls in this
        # fixture → B's contribution drops, A's survives via cf signal
        # only — but since cf weight is 0 too, both end at 0 → fallback
        # to equal split. Verify that fallback behaviour kicks in.
        s, task_id = _build_resp_scenario()
        cfg = XAIScoringConfig(
            responsibility_weights={
                "counterfactual": 0.0, "tool_impact": 1.0,
                "message_efficacy": 0.0, "memory_used": 0.0,
                "usefulness": 0.0, "causal_centrality": 0.0,
            },
        )
        gen = AccountabilityReportGenerator(
            store=s, pipeline=_MockPipeline({"final_diagnosis": "Y",
                                              "confidence": 0.5}),
            llm=_FakeLLM(), config=cfg,
        )
        report = gen.generate(task_id)
        # Neither specialist has tools → both contributing-zero → fallback.
        assert sum(report.agent_responsibility_scores.values()) == pytest.approx(1.0)


class TestFallbackExplanationConfig:
    def test_default_epsilon_groups_close(self):
        r = AccountabilityReport(
            final_outcome="MI", outcome_correct=True,
            agent_responsibility_scores={"specialist_a": 0.50, "specialist_b": 0.49},
        )
        s = _fallback_explanation(r)
        # Default epsilon = 0.05 → both agents tied → "shared" credit.
        assert "shared" in s.lower() or " and " in s

    def test_custom_epsilon_singles_out(self):
        r = AccountabilityReport(
            final_outcome="MI", outcome_correct=True,
            agent_responsibility_scores={"specialist_a": 0.50, "specialist_b": 0.49},
        )
        cfg = XAIScoringConfig(tie_epsilon=0.001)
        s = _fallback_explanation(r, config=cfg)
        # Tighter epsilon → not tied → only A is credited.
        assert "specialist_a" in s
        assert "specialist_b" not in s
