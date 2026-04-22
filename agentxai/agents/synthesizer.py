"""
Synthesizer — reads both specialists' memories and produces the final
diagnosis. No tools are called; this agent's only side effect (besides
log_action / memory writes) is one LLM call to Gemini.

Pipeline (one ``active_plan``):

1. ``read_specialist_memories`` — snapshot Specialist A and B's memory dicts
2. ``synthesize_diagnosis``     — single Gemini call with strict JSON contract

Returns and writes to memory:

    {
        "final_diagnosis": str,
        "confidence":      float,        # clamped to [0, 1]
        "differential":    List[str],    # alternative diagnoses, ranked
        "rationale":       str,
    }

Defensive parsing: if the LLM response cannot be parsed as JSON we still
return all four keys (``final_diagnosis=""``, ``confidence=0.0``,
``differential=[]``, ``rationale="<raw response>"``) so downstream callers
never have to special-case missing fields.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from agentxai.agents._llm_utils import extract_text, parse_json_object
from agentxai.agents.base import TracedAgent, make_default_llm


_SYNTH_PROMPT = (
    "You are a senior physician synthesizing a final diagnosis from two "
    "junior specialists' findings.\n\n"
    "Patient case:\n{case}\n\n"
    "Specialist A — Symptom Analyzer (memory dump):\n{mem_a}\n\n"
    "Specialist B — Evidence Retriever (memory dump):\n{mem_b}\n\n"
    "Answer-option choices (if any):\n{options}\n\n"
    "Return ONLY a single JSON object with EXACTLY these keys, no extra text:\n"
    '  "final_diagnosis": string  (the most likely diagnosis; if answer-option '
    'choices are provided, use the exact text of the chosen option),\n'
    '  "confidence": number between 0 and 1,\n'
    '  "differential": array of strings (ranked alternative diagnoses),\n'
    '  "rationale": string (2–4 sentences citing the specialists\' findings)\n'
)


class Synthesizer(TracedAgent):
    """Final-diagnosis synthesizer over both specialists' memory states."""

    def __init__(
        self,
        *,
        agent_id: str = "synthesizer",
        specialist_a_id: str = "specialist_a",
        specialist_b_id: str = "specialist_b",
        llm: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            llm=llm if llm is not None else make_default_llm(),
            **kwargs,
        )
        self.specialist_a_id = specialist_a_id
        self.specialist_b_id = specialist_b_id

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self, input_payload: dict) -> dict:
        case = (
            input_payload.get("patient_case")
            or input_payload.get("question")
            or ""
        )
        options = input_payload.get("options") or input_payload.get("answer_options") or {}

        with self.active_plan([
            "read_specialist_memories",
            "synthesize_diagnosis",
        ]):
            # 1. Snapshot both specialists' memories (use dict.items() to avoid
            #    triggering LoggedMemory.__getitem__ — we want one trajectory
            #    event per snapshot, not one per key).
            mem_a = self._snapshot_memory(self.specialist_a_id)
            mem_b = self._snapshot_memory(self.specialist_b_id)
            self.log_action(
                "read_specialist_memories",
                {
                    "specialist_a_keys": sorted(mem_a.keys()),
                    "specialist_b_keys": sorted(mem_b.keys()),
                },
                outcome=f"a:{len(mem_a)} keys, b:{len(mem_b)} keys",
            )

            # 2. Single Gemini call with strict JSON contract
            result = self._synthesize(case, options, mem_a, mem_b)
            self.log_action(
                "synthesize_diagnosis",
                {"case_chars": len(case)},
                outcome=result.get("final_diagnosis", "") or "no diagnosis",
            )

            self.memory["final_output"] = result

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _snapshot_memory(self, agent_id: str) -> Dict[str, Any]:
        mem = self.memory_logger.for_agent(agent_id)
        return {k: v for k, v in mem.items()}

    def _synthesize(
        self,
        case: str,
        options: Dict[str, str],
        mem_a: Dict[str, Any],
        mem_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.llm is None:
            return _empty_result("LLM unavailable.")

        prompt = _SYNTH_PROMPT.format(
            case=case,
            mem_a=json.dumps(mem_a, indent=2, default=str),
            mem_b=json.dumps(mem_b, indent=2, default=str),
            options=json.dumps(options, indent=2),
        )

        try:
            response = self.llm.invoke(prompt)
        except Exception as exc:
            return _empty_result(f"LLM error: {type(exc).__name__}: {exc}")

        raw = extract_text(response)
        parsed = parse_json_object(raw)
        return _normalise_result(parsed, raw)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _empty_result(rationale: str) -> Dict[str, Any]:
    return {
        "final_diagnosis": "",
        "confidence":      0.0,
        "differential":    [],
        "rationale":       rationale,
    }


def _normalise_result(parsed: Any, raw: str) -> Dict[str, Any]:
    """Coerce whatever came back into the four-key contract."""
    if not isinstance(parsed, dict):
        return _empty_result(raw.strip())

    final = str(parsed.get("final_diagnosis", "") or "").strip()
    rationale = str(parsed.get("rationale", "") or "").strip()

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    differential_raw = parsed.get("differential", []) or []
    if not isinstance(differential_raw, list):
        differential_raw = [differential_raw]
    differential: List[str] = [str(x).strip() for x in differential_raw if str(x).strip()]

    return {
        "final_diagnosis": final,
        "confidence":      confidence,
        "differential":    differential,
        "rationale":       rationale or raw.strip(),
    }
