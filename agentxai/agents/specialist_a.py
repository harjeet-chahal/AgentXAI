"""
Specialist A — Symptom Analyzer.

Pipeline (one ``active_plan``):

1. ``extract_symptoms``   — LLM extracts symptom phrases from the patient case
2. ``lookup_conditions``  — symptom_lookup() per phrase, aggregate likelihoods
3. ``score_severity``     — severity_scorer() over the extracted symptoms
4. ``summarize_findings`` — write the four canonical memory keys and notify
                            the Orchestrator via a ``finding`` message

Memory keys written (LoggedMemory writes are auto-traced as MemoryDiffs):

    symptom_patterns : List[str]
    severity_score   : float                # 0–1 from severity_scorer
    top_conditions   : List[Tuple[str, float]]   # (condition, agg_likelihood)
    confidence       : float                # share of symptoms with hits
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from agentxai.agents._llm_utils import extract_text, parse_json_list
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

        with self.active_plan([
            "extract_symptoms",
            "lookup_conditions",
            "score_severity",
            "summarize_findings",
        ]):
            # 1. Extract symptoms
            symptoms = self._extract_symptoms(case)
            self.log_action(
                "extract_symptoms",
                {"case_chars": len(case)},
                outcome=f"{len(symptoms)} symptoms",
            )

            # 2. symptom_lookup per phrase, aggregate likelihoods
            agg: Dict[str, float] = defaultdict(float)
            hits = 0
            for s in symptoms:
                result = self.symptom_lookup_fn(s) or {}
                related = result.get("related_conditions") or []
                if related:
                    hits += 1
                for cond, like in related:
                    agg[cond] += float(like)
            top_conditions: List[Tuple[str, float]] = [
                (c, round(score, 4))
                for c, score in sorted(agg.items(), key=lambda x: (-x[1], x[0]))[: self.top_k]
            ]
            self.log_action(
                "lookup_conditions",
                {"n_symptoms": len(symptoms), "n_hits": hits},
                outcome=f"{len(top_conditions)} candidate conditions",
            )

            # 3. Severity
            severity = float(self.severity_scorer_fn(symptoms) or 0.0)
            self.log_action(
                "score_severity",
                {"symptoms": symptoms},
                outcome=f"severity={severity:.3f}",
            )

            # 4. Summarise → memory + finding message.
            # Wrap the writes + message in `traced_action` so each MemoryDiff
            # links to the summarize_findings event (otherwise the writes
            # fire before any event exists and land "outside_traced_action").
            confidence = (hits / len(symptoms)) if symptoms else 0.0
            findings: Dict[str, Any] = {
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
