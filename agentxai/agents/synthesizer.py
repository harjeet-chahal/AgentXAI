"""
Synthesizer — reads both specialists' memories and produces the final
MedQA option choice. No tools are called; this agent's only side effect
(besides log_action / memory writes) is one LLM call to Gemini.

Pipeline (one ``active_plan``):

1. ``read_specialist_memories`` — snapshot Specialist A and B's memory dicts
2. ``synthesize_diagnosis``     — single LLM call with strict JSON contract

Returns and writes to memory the following keys (the per-option fields
were added so the rationale must be grounded in the listed options
rather than the LLM's prior training, which sometimes drifts to outdated
guidelines — e.g., quoting Western blot when the listed option is the
modern HIV-1/HIV-2 antibody differentiation immunoassay):

    {
        "final_diagnosis":          str,            # mirrors predicted_text
        "predicted_letter":         str,            # "A".."E", "" if unknown
        "predicted_text":           str,            # verbatim option text
        "confidence":               float,          # clamped to [0, 1]
        "differential":             List[str],      # ranked alternatives
        "rationale":                str,            # explains *why* > others
        "option_analysis":          List[OptionVerdict],
        "supporting_evidence_ids":  List[str],
    }

Where each ``OptionVerdict`` is::

    {
        "letter": "A",
        "text":   "Western blot",
        "verdict": "correct" | "incorrect" | "partial",
        "reason":  "Western blot was the older confirmatory test; ..."
    }

Defensive parsing: every new field is OPTIONAL — older LLM responses (or
records produced before this change) that only carry the original four
keys still parse cleanly. Missing fields default to empty / [] / 0.0;
malformed entries are dropped rather than rejected wholesale.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from agentxai.agents._llm_utils import extract_text, parse_json_object
from agentxai.agents.base import TracedAgent, make_default_llm


_SYNTH_PROMPT = (
    "You are a senior physician answering a USMLE-style multiple-choice "
    "question by synthesizing two junior specialists' findings.\n\n"
    "Patient case:\n{case}\n\n"
    "Specialist A — Symptom Analyzer (memory dump):\n{mem_a}\n\n"
    "Specialist B — Evidence Retriever (memory dump):\n{mem_b}\n\n"
    "Answer options:\n{options}\n\n"
    "ANSWERING RULES (read carefully — many wrong answers come from "
    "ignoring these):\n"
    "  * Choose ONE of the listed options. Do NOT recommend a "
    "test/treatment/diagnosis that is not listed unless you are explaining "
    "why a *listed* option is incorrect.\n"
    "  * If your prior training conflicts with the listed options "
    "(e.g., you remember an older guideline that names a test not in "
    "the options), defer to the listed option text. Modern question "
    "banks reflect current guidelines — the option set is the ground "
    "truth for what the answer space looks like.\n"
    "  * Your rationale MUST justify why the chosen option is better "
    "than each alternative — not just why it fits the case in isolation.\n"
    "  * Cite supporting_evidence_ids using the exact `doc_id` values "
    "from Specialist B's `top_evidence` / `retrieved_docs`. Skip the "
    "field if no doc is genuinely supporting.\n\n"
    "Return ONLY a single JSON object with EXACTLY these keys, no extra "
    "text, no markdown fences:\n"
    '  "predicted_letter": string  (one of the option letters, e.g. "C"),\n'
    '  "predicted_text":   string  (verbatim text of the chosen option),\n'
    '  "final_diagnosis":  string  (same as predicted_text — kept for '
    'backward compatibility),\n'
    '  "confidence":       number between 0 and 1,\n'
    '  "differential":     array of strings (ranked alternative '
    'diagnoses, may overlap with the option list),\n'
    '  "rationale":        string (2-4 sentences explaining why '
    'predicted_letter is better than every other option),\n'
    '  "option_analysis":  array of objects, ONE PER OPTION, each with:\n'
    '       {{"letter": "A", "text": "<verbatim option text>", '
    '"verdict": "correct" | "incorrect" | "partial", '
    '"reason": "<one sentence>"}},\n'
    '  "supporting_evidence_ids": array of strings (doc_ids from '
    "Specialist B; may be empty)\n"
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

            # 2. Single LLM call with strict JSON contract.
            # Wrap the memory write in `traced_action` so the resulting
            # MemoryDiff for `final_output` links to this event rather than
            # landing "outside_traced_action".
            result = self._synthesize(case, options, mem_a, mem_b)
            with self.traced_action(
                "synthesize_diagnosis",
                {"case_chars": len(case)},
                outcome=result.get("final_diagnosis", "") or "no diagnosis",
            ):
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

_VALID_VERDICTS: tuple = ("correct", "incorrect", "partial")


def _empty_result(rationale: str) -> Dict[str, Any]:
    return {
        "final_diagnosis":         "",
        "predicted_letter":        "",
        "predicted_text":          "",
        "confidence":              0.0,
        "differential":            [],
        "rationale":               rationale,
        "option_analysis":         [],
        "supporting_evidence_ids": [],
    }


def _normalise_letter(value: Any) -> str:
    """Single uppercase A-Z letter, or empty string if value is unusable."""
    if value is None:
        return ""
    s = str(value).strip().upper()
    if len(s) == 1 and "A" <= s <= "Z":
        return s
    return ""


def _normalise_verdict(value: Any) -> str:
    """One of `correct` / `incorrect` / `partial`; else empty."""
    s = str(value or "").strip().lower()
    return s if s in _VALID_VERDICTS else ""


def _normalise_option_analysis(raw: Any) -> List[Dict[str, Any]]:
    """
    Coerce option_analysis into a list of well-formed entries.

    Each entry is `{letter, text, verdict, reason}`. Bad entries are
    silently dropped rather than blocking the whole record — older
    responses that omit option_analysis entirely simply yield [].
    """
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    seen_letters: set = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        letter = _normalise_letter(item.get("letter"))
        if not letter or letter in seen_letters:
            continue
        text = str(item.get("text") or "").strip()
        verdict = _normalise_verdict(item.get("verdict"))
        reason = str(item.get("reason") or "").strip()
        seen_letters.add(letter)
        out.append({
            "letter":  letter,
            "text":    text,
            "verdict": verdict,
            "reason":  reason,
        })
    out.sort(key=lambda x: x["letter"])
    return out


def _normalise_evidence_ids(raw: Any) -> List[str]:
    """List[str], deduped, non-empty entries only."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen: set = set()
    for item in raw:
        s = str(item or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _normalise_result(parsed: Any, raw: str) -> Dict[str, Any]:
    """
    Coerce whatever came back into the full eight-key contract.

    Backward compatibility: any subset of the eight keys is accepted —
    `predicted_letter` / `predicted_text` / `option_analysis` /
    `supporting_evidence_ids` were added later and an LLM that emits the
    older four-key shape (`final_diagnosis`, `confidence`, `differential`,
    `rationale`) still parses fine. `predicted_text` falls back to
    `final_diagnosis` and vice versa so callers can read either key.
    """
    if not isinstance(parsed, dict):
        return _empty_result(raw.strip())

    final = str(parsed.get("final_diagnosis", "") or "").strip()
    predicted_text = str(parsed.get("predicted_text", "") or "").strip()
    # Mutually-fall-back so old responses still populate predicted_text.
    if not predicted_text and final:
        predicted_text = final
    if not final and predicted_text:
        final = predicted_text

    predicted_letter = _normalise_letter(parsed.get("predicted_letter"))
    rationale = str(parsed.get("rationale", "") or "").strip()

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    differential_raw = parsed.get("differential", []) or []
    if not isinstance(differential_raw, list):
        differential_raw = [differential_raw]
    differential: List[str] = [
        str(x).strip() for x in differential_raw if str(x).strip()
    ]

    option_analysis = _normalise_option_analysis(parsed.get("option_analysis"))
    supporting = _normalise_evidence_ids(parsed.get("supporting_evidence_ids"))

    # If the LLM gave option_analysis but no top-level predicted_letter,
    # backfill from the option marked "correct" so the pipeline doesn't
    # fall back to text matching unnecessarily.
    if not predicted_letter:
        for entry in option_analysis:
            if entry["verdict"] == "correct":
                predicted_letter = entry["letter"]
                if not predicted_text and entry["text"]:
                    predicted_text = entry["text"]
                if not final:
                    final = predicted_text
                break

    return {
        "final_diagnosis":         final,
        "predicted_letter":        predicted_letter,
        "predicted_text":          predicted_text,
        "confidence":              confidence,
        "differential":            differential,
        "rationale":               rationale or raw.strip(),
        "option_analysis":         option_analysis,
        "supporting_evidence_ids": supporting,
    }
