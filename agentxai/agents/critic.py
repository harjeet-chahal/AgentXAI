"""
Critic — self-critique pass over the Synthesizer's final output.

The Critic is invoked AFTER the Synthesizer produces its draft answer but
BEFORE the pipeline persists ``system_output`` and runs the XAI layer. Its
job is to spot weaknesses in the chosen answer (missing differentials,
ignored evidence, logical gaps) so the Pipeline can decide whether to
re-call the Orchestrator with the gaps injected as feedback.

Pipeline (one ``active_plan``):

1. ``critique_synthesis`` — single LLM call asking the model to review the
   rationale + predicted_letter against the answer options and return a
   strict-JSON judgement.
2. The result is written to memory under the ``critique`` key so it shows
   up in the dashboard's memory tab on a dedicated ``critic`` agent_id.

``run()`` returns::

    {
        "needs_revision":          bool,
        "missing_considerations":  List[str],
        "confidence_in_critique":  float,   # 0..1
    }

Defensive parsing: any malformed / missing keys collapse to a "no
revision needed" verdict so a critic outage never blocks the pipeline.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from agentxai.agents._llm_utils import extract_text, parse_json_object
from agentxai.agents.base import TracedAgent, make_default_llm


_CRITIC_PROMPT = (
    "You are a senior physician auditor. Another model just answered a "
    "USMLE-style multiple-choice question. Your ONLY job is to detect "
    "errors that would change the predicted letter — not to grade prose "
    "quality, completeness, or thoroughness.\n\n"
    "Answer options:\n{options}\n\n"
    "Predicted letter: {predicted_letter}\n"
    "Predicted text:   {predicted_text}\n\n"
    "Rationale from the synthesizer:\n{rationale}\n\n"
    "Decision rule — read carefully:\n"
    "  Default to needs_revision=false. Only set needs_revision=true if "
    "you can name a SPECIFIC differential or piece of evidence the "
    "rationale missed that would PLAUSIBLY flip the predicted letter to "
    "a different option. Minor weaknesses, stylistic gaps, missing "
    "background detail, or 'could be more thorough' do NOT qualify — if "
    "the predicted letter would still be correct after addressing the "
    "gap, return needs_revision=false.\n"
    "  Do NOT flag revision merely because: the rationale is brief, a "
    "distractor wasn't explicitly discussed, the differential list is "
    "short, or the wording could be improved. The bar is: 'this gap "
    "would change the answer'.\n\n"
    "Return ONLY a single JSON object with EXACTLY these keys, no markdown, "
    "no extra prose:\n"
    '  "needs_revision":         bool   (true ONLY when a named gap would '
    "plausibly flip the predicted letter),\n"
    '  "missing_considerations": array of short strings, each naming a '
    "specific differential/evidence gap that could change the answer "
    "(empty when needs_revision is false),\n"
    '  "confidence_in_critique": number in [0, 1] expressing how sure you '
    "are about your verdict.\n"
)


class Critic(TracedAgent):
    """LLM-driven self-critique pass over the Synthesizer's output."""

    def __init__(
        self,
        *,
        agent_id: str = "critic",
        llm: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            llm=llm if llm is not None else make_default_llm(),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self, input_payload: dict) -> dict:
        rationale = str(input_payload.get("rationale", "") or "")
        predicted_letter = str(input_payload.get("predicted_letter", "") or "")
        predicted_text = str(input_payload.get("predicted_text", "") or "")
        options = dict(input_payload.get("options") or {})

        with self.active_plan(["critique_synthesis"]):
            critique = self._critique(
                rationale=rationale,
                predicted_letter=predicted_letter,
                predicted_text=predicted_text,
                options=options,
            )
            with self.traced_action(
                "critique_synthesis",
                {
                    "predicted_letter": predicted_letter,
                    "n_options":        len(options),
                    "rationale_chars":  len(rationale),
                },
                outcome=(
                    "needs_revision="
                    f"{critique['needs_revision']}, "
                    f"{len(critique['missing_considerations'])} gaps"
                ),
            ):
                self.memory["critique"] = critique
                self.memory["missing_considerations"] = list(
                    critique["missing_considerations"]
                )

        return critique

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _critique(
        self,
        *,
        rationale: str,
        predicted_letter: str,
        predicted_text: str,
        options: Dict[str, str],
    ) -> Dict[str, Any]:
        if self.llm is None:
            return _empty_critique()

        prompt = _CRITIC_PROMPT.format(
            options=json.dumps(options, indent=2),
            predicted_letter=predicted_letter or "(unset)",
            predicted_text=predicted_text or "(unset)",
            rationale=rationale or "(empty)",
        )

        try:
            response = self.llm.invoke(prompt)
        except Exception:
            return _empty_critique()

        parsed = parse_json_object(extract_text(response))
        return _normalise_critique(parsed)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _empty_critique() -> Dict[str, Any]:
    return {
        "needs_revision":         False,
        "missing_considerations": [],
        "confidence_in_critique": 0.0,
    }


def _normalise_critique(parsed: Any) -> Dict[str, Any]:
    """Coerce arbitrary LLM JSON into the strict three-key contract."""
    if not isinstance(parsed, dict):
        return _empty_critique()

    needs = parsed.get("needs_revision", False)
    if isinstance(needs, str):
        needs = needs.strip().lower() in ("true", "yes", "1")
    needs = bool(needs)

    raw_missing = parsed.get("missing_considerations", []) or []
    if isinstance(raw_missing, str):
        raw_missing = [raw_missing]
    if not isinstance(raw_missing, list):
        raw_missing = []
    missing: List[str] = [
        str(item).strip() for item in raw_missing if str(item).strip()
    ]

    try:
        confidence = float(parsed.get("confidence_in_critique", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    # If the LLM reports a gap list while marking needs_revision=False,
    # honor the explicit boolean — the gaps are noted but not actionable.
    return {
        "needs_revision":         needs,
        "missing_considerations": missing,
        "confidence_in_critique": confidence,
    }
