"""
Centralised scoring configuration for the XAI runtime.

Every numeric knob the causal DAG, counterfactual engine, and
accountability scorer use lives here as a field on
:class:`XAIScoringConfig`. The defaults reproduce the historical
behaviour exactly — instantiating ``XAIScoringConfig()`` with no
arguments and passing it to the three component constructors gives
identical scoring to the pre-config code, so adopting the config is a
no-op for existing tests, stored records, and downstream consumers.

These weights are heuristic
--------------------------
**The weights here are HEURISTIC. They were calibrated qualitatively
against the test fixtures in ``tests/`` — there is no held-out scoring
rubric, no ground-truth labels, and no formal calibration study.**

Tuning is an open research question:

  * Counterfactual delta weights (``cf_dx_weight`` + ``cf_conf_weight``)
    were chosen so a diagnosis flip alone yields ≈0.6 and a confidence
    swing alone tops out around ≈0.4 — never above the diagnosis-flip
    floor. A formal ablation against a labelled outcome-change set could
    re-tune them.
  * Causal-DAG edge weights are gut-feel: temporal precedence weakest,
    acted-upon messages strongest, ignored messages roughly half. A
    sensitivity sweep over downstream accountability scores would
    surface whether the ratio matters.
  * Responsibility-composite weights were chosen to keep the
    counterfactual signal dominant (0.35) without letting it eclipse
    structural signals (tool/message/memory). Sensible-default territory.
  * Question-type priors are direction-only — they only encode "for a
    pharmacology question, A's symptom analysis matters less than B's
    drug knowledge". The exact 0.5/0.6/0.7 values are unstudied.

**Pass a customised** ``XAIScoringConfig`` **instance to** ``CausalDAGBuilder``,
``CounterfactualEngine``, **and** ``AccountabilityReportGenerator`` **to
ablate.** A planned next step is a sweep harness in ``eval/`` that
varies one knob at a time and reports the deltas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# Defaults extracted to module-level dicts so they're easy to re-import
# (e.g., in tests that want to verify the historical values without
# instantiating a config).

_DEFAULT_RESPONSIBILITY_WEIGHTS: Dict[str, float] = {
    "counterfactual":    0.35,
    "tool_impact":       0.20,
    "message_efficacy":  0.15,
    "memory_used":       0.15,
    "usefulness":        0.10,
    "causal_centrality": 0.05,
}

_DEFAULT_QUESTION_TYPE_PRIORS: Dict[str, Dict[str, float]] = {
    "diagnosis":         {"specialist_a": 1.0, "specialist_b": 1.0},
    "treatment":         {"specialist_a": 0.7, "specialist_b": 1.0},
    "screening_or_test": {"specialist_a": 0.5, "specialist_b": 1.0},
    "pharmacology":      {"specialist_a": 0.6, "specialist_b": 1.0},
    "mechanism":         {"specialist_a": 0.7, "specialist_b": 1.0},
    "risk_factor":       {"specialist_a": 0.8, "specialist_b": 1.0},
    "anatomy":           {"specialist_a": 0.5, "specialist_b": 1.0},
    "prognosis":         {"specialist_a": 0.7, "specialist_b": 1.0},
    "unknown":           {"specialist_a": 1.0, "specialist_b": 1.0},
}


@dataclass
class XAIScoringConfig:
    """All numeric knobs used by the causal/counterfactual/accountability stack.

    Defaults reproduce historical behaviour. Pass a customised instance
    to the three component constructors to ablate.
    """

    # ------------------------------------------------------------------
    # Counterfactual outcome delta
    # ------------------------------------------------------------------
    # ``_outcome_delta(orig, perturbed)`` =
    #     cf_dx_weight  * (1 if final_diagnosis flipped else 0)
    #   + cf_conf_weight * |orig.confidence - perturbed.confidence|
    # The weights deliberately sum to 1.0 so a complete flip + complete
    # confidence inversion saturates at exactly 1.0.
    cf_dx_weight: float = 0.6
    cf_conf_weight: float = 0.4

    # ------------------------------------------------------------------
    # Causal DAG edge weights
    # ------------------------------------------------------------------
    # Within-agent temporal precedence — the weakest edge type, marked
    # ``causal_type="contributory"``. Used as the base weight for every
    # consecutive pair of events from the same agent.
    temporal_edge_weight: float = 0.3
    # Inter-agent message edges — used when no counterfactual delta is
    # available for the message in the ``counterfactual_runs`` table.
    # The cf delta always overrides these heuristics when present.
    message_acted_upon_weight: float = 1.0
    message_ignored_weight: float = 0.5

    # ------------------------------------------------------------------
    # Root-cause selection (accountability.py::_select_root_cause)
    # ------------------------------------------------------------------
    # Bonuses added on top of an ancestor's outgoing-graph-weight base
    # score before the upstream factor is applied. Tool impact dominates
    # because high-impact tool calls are almost always the meaningful
    # upstream cause; messages and memory writes are smaller boosters.
    root_tool_bonus: float = 1.5
    root_acted_msg_bonus: float = 1.0
    root_mem_bonus: float = 0.5
    # Maximum fraction by which the latest candidate is shaved relative
    # to the earliest. Set small (0.3 → at most 30% discount) so impact
    # still dominates ordering; the upstream factor is only a tie-breaker
    # for near-equal scores.
    root_upstream_discount: float = 0.3

    # ------------------------------------------------------------------
    # Accountability composite
    # ------------------------------------------------------------------
    # Composite weight per per-agent signal in ``_combine_signals``.
    # Must sum to ~1.0 so the unnormalised score lands in [0, 1] when
    # every signal is in [0, 1]. See ``_compute_responsibility_signals``
    # for the per-signal definitions.
    responsibility_weights: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_RESPONSIBILITY_WEIGHTS)
    )
    # Per-(question_type, agent_id) prior multiplier applied to the
    # composite score before final normalisation. Values < 1.0 shrink an
    # agent's share for question types where its work is less relevant
    # (e.g., Specialist A on a pharmacology question). See
    # ``agentxai/data/question_classifier.py`` for the type taxonomy.
    question_type_priors: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            qt: dict(agents)
            for qt, agents in _DEFAULT_QUESTION_TYPE_PRIORS.items()
        }
    )
    # Two agents whose normalised responsibility scores are within
    # ``tie_epsilon`` of each other are treated as tied: the explanation
    # credits both rather than singling one out. Used by the LLM prompt
    # builder and the templated fallback.
    tie_epsilon: float = 0.05


# Module-level default instance. Components fall back to this when no
# explicit config is passed — preserves the simple-call-site ergonomics
# of the pre-config code.
DEFAULT_CONFIG: XAIScoringConfig = XAIScoringConfig()
