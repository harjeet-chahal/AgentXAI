"""
Specialist A — Symptom Analyzer.

Pipeline (one ``active_plan``):

1. ``extract_symptoms``   — LLM extracts symptom phrases from the patient case
2. ``lookup_conditions``  — LLM picks the symptom subset that benefits most
                            from symptom_lookup, then runs symptom_lookup()
                            on each chosen phrase and aggregates likelihoods
3. ``score_severity``     — LLM decides whether severity_scorer() is
                            informative for this case; only runs the tool
                            when the LLM returns ``run: true``
4. ``summarize_findings`` — write the four canonical memory keys and notify
                            the Orchestrator via a ``finding`` message

Tool selection in steps 2 and 3 is LLM-driven so that Pillar 3 (Tool
Provenance) varies by case rather than recording the same fixed call
sequence every run. The LLM's reasoning is recorded as the trajectory
event's ``outcome`` field so it surfaces in the dashboard.

Memory keys written (LoggedMemory writes are auto-traced as MemoryDiffs):

    symptom_patterns : List[str]
    severity_score   : float                # 0–1 from severity_scorer
    top_conditions   : List[Tuple[str, float]]   # (condition, agg_likelihood)
    confidence       : float                # share of symptoms with hits
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from agentxai.agents._llm_utils import (
    extract_text,
    parse_json_list,
    parse_json_object,
)
from agentxai.agents.base import TracedAgent, make_default_llm
from agentxai.tools.severity_scorer import severity_scorer as _real_severity_scorer
from agentxai.tools.symptom_lookup import symptom_lookup as _real_symptom_lookup


_SYMPTOM_PROMPT = (
    "You are a clinical NLP extractor. From the patient case below, list the "
    "presenting symptoms as short, lower-case phrases (e.g. 'chest pain', "
    "'shortness of breath').\n"
    "Return ONLY a JSON array of strings. No prose, no markdown.\n\n"
    "Patient case:\n{case}"
)

_LOOKUP_SUBSET_PROMPT = (
    "Given these symptoms {symptoms}, which subset would benefit most from "
    "symptom_lookup? Return JSON list of phrases."
)

_SEVERITY_DECISION_PROMPT = (
    "Should we score severity for this case? Return JSON "
    '{{"run": bool, "reason": str}}.\n\n'
    "Patient case:\n{case}\n"
    "Extracted symptoms: {symptoms}"
)


class SpecialistA(TracedAgent):
    """Symptom analyzer — symptom_lookup + severity_scorer + LLM extraction."""

    def __init__(
        self,
        *,
        agent_id: str = "specialist_a",
        symptom_lookup_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        severity_scorer_fn: Optional[Callable[[List[str]], float]] = None,
        orchestrator_id: str = "orchestrator",
        top_k: int = 5,
        llm: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            llm=llm if llm is not None else make_default_llm(),
            **kwargs,
        )
        self.symptom_lookup_fn = symptom_lookup_fn or _real_symptom_lookup
        self.severity_scorer_fn = severity_scorer_fn or _real_severity_scorer
        self.orchestrator_id = orchestrator_id
        self.top_k = top_k

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self, input_payload: dict) -> dict:
        case = (
            input_payload.get("patient_case")
            or input_payload.get("question")
            or ""
        )

        available_actions: List[str] = [
            "extract_symptoms",
            "lookup_conditions",
            "score_severity",
            "summarize_findings",
        ]
        plan = self.generate_plan(case, available_actions)

        symptoms: List[str] = []
        top_conditions: List[Tuple[str, float]] = []
        hits = 0
        severity = 0.0
        findings: Dict[str, Any] = {}

        with self.active_plan(plan):
            for action in plan:
                if action == "extract_symptoms":
                    symptoms = self._extract_symptoms(case)
                    self.log_action(
                        "extract_symptoms",
                        {"case_chars": len(case)},
                        outcome=f"{len(symptoms)} symptoms",
                    )
                elif action == "lookup_conditions":
                    subset, reasoning = self._select_lookup_subset(symptoms)
                    agg: Dict[str, float] = defaultdict(float)
                    for s in subset:
                        result = self.symptom_lookup_fn(s) or {}
                        related = result.get("related_conditions") or []
                        if related:
                            hits += 1
                        for cond, like in related:
                            agg[cond] += float(like)
                    top_conditions = [
                        (c, round(score, 4))
                        for c, score in sorted(
                            agg.items(), key=lambda x: (-x[1], x[0])
                        )[: self.top_k]
                    ]
                    outcome = (
                        f"LLM picked {len(subset)}/{len(symptoms)} symptoms; "
                        f"{len(top_conditions)} candidate conditions"
                    )
                    if reasoning:
                        outcome = f"{outcome} | reasoning: {reasoning}"
                    self.log_action(
                        "lookup_conditions",
                        {
                            "n_symptoms":  len(symptoms),
                            "n_picked":    len(subset),
                            "picked":      subset,
                            "n_hits":      hits,
                        },
                        outcome=outcome,
                    )
                elif action == "score_severity":
                    run_it, reason = self._decide_score_severity(case, symptoms)
                    if run_it:
                        severity = float(self.severity_scorer_fn(symptoms) or 0.0)
                        outcome = f"severity={severity:.3f}"
                        if reason:
                            outcome = f"{outcome} | reasoning: {reason}"
                    else:
                        outcome = "skipped by LLM"
                        if reason:
                            outcome = f"{outcome} | reasoning: {reason}"
                    self.log_action(
                        "score_severity",
                        {
                            "symptoms":  symptoms,
                            "ran":       run_it,
                        },
                        outcome=outcome,
                    )
                elif action == "summarize_findings":
                    # Wrap the writes + message in `traced_action` so each
                    # MemoryDiff links to the summarize_findings event
                    # (otherwise writes fire before any event exists and
                    # land "outside_traced_action").
                    confidence = (hits / len(symptoms)) if symptoms else 0.0
                    findings = {
                        "symptom_patterns": symptoms,
                        "severity_score":   severity,
                        "top_conditions":   top_conditions,
                        "confidence":       round(confidence, 4),
                    }
                    with self.traced_action(
                        "summarize_findings",
                        {"keys": list(findings.keys())},
                        outcome="memory+message written",
                    ):
                        for k, v in findings.items():
                            self.memory[k] = v

                        self.send_message(
                            receiver=self.orchestrator_id,
                            message_type="finding",
                            content=findings,
                        )

        return findings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_symptoms(self, case: str) -> List[str]:
        """LLM-driven symptom extraction; empty list if no LLM is wired."""
        if not case or self.llm is None:
            return []
        try:
            response = self.llm.invoke(_SYMPTOM_PROMPT.format(case=case))
        except Exception:
            return []
        return parse_json_list(extract_text(response))

    def _select_lookup_subset(
        self,
        symptoms: List[str],
    ) -> Tuple[List[str], str]:
        """
        Ask the LLM which symptoms benefit most from symptom_lookup.

        Returns ``(chosen_subset, raw_reasoning_text)``. Falls back to the
        full ``symptoms`` list when:
          - no LLM is wired or no symptoms were extracted,
          - the LLM raises or returns junk,
          - the LLM returns names that don't match the extracted symptoms.

        The fallback preserves the pre-LLM-gating behaviour so an LLM
        outage degrades to "look up everything" rather than nothing.
        """
        if not symptoms or self.llm is None:
            return list(symptoms), ""
        try:
            response = self.llm.invoke(
                _LOOKUP_SUBSET_PROMPT.format(symptoms=symptoms),
            )
        except Exception:
            return list(symptoms), ""

        text = extract_text(response)
        parsed = parse_json_list(text)
        if not parsed:
            return list(symptoms), text

        symptom_set = {s.lower() for s in symptoms}
        chosen = [s for s in parsed if s.lower() in symptom_set]
        if not chosen:
            return list(symptoms), text
        return chosen, text

    def _decide_score_severity(
        self,
        case: str,
        symptoms: List[str],
    ) -> Tuple[bool, str]:
        """
        Ask the LLM whether severity scoring should run for this case.

        Returns ``(run_it, reason)``. Falls back to ``(True, "")`` when no
        LLM is wired, the LLM raises, or the response can't be parsed as
        ``{"run": bool, "reason": str}`` — same default-on-failure stance
        as ``_select_lookup_subset``.
        """
        if self.llm is None:
            return True, ""
        try:
            response = self.llm.invoke(
                _SEVERITY_DECISION_PROMPT.format(case=case, symptoms=symptoms),
            )
        except Exception:
            return True, ""

        parsed = parse_json_object(extract_text(response))
        if not parsed:
            return True, ""

        run_value = parsed.get("run")
        reason = str(parsed.get("reason", "") or "")
        if isinstance(run_value, bool):
            return run_value, reason
        # Non-bool → treat as missing, fall back to running.
        return True, reason
