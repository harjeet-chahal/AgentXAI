"""
Tiny helpers for parsing LangChain LLM responses inside the agents package.

Kept private (underscore-prefixed module) and dependency-light so agents and
unit tests can both import them without dragging in the whole LangChain stack.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_text(response: Any) -> str:
    """
    Normalise whatever a chat model returned (string, AIMessage, list of
    Anthropic content blocks, etc.) into a plain string.
    """
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def parse_json_list(text: str) -> List[str]:
    """
    Defensively extract a JSON list of strings from *text*. Tries (in order):
    full JSON parse, embedded ``[...]`` substring, then a line-based fallback.
    Returns ``[]`` if nothing parseable is found.
    """
    if not text:
        return []
    stripped = text.strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    match = _ARRAY_RE.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

    return [
        ln.strip(" -*\t•")
        for ln in text.splitlines()
        if ln.strip(" -*\t•")
    ]


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Defensively extract a JSON object from *text*. Tries full parse, then any
    embedded ``{...}`` substring. Returns ``None`` if nothing parses.
    """
    if not text:
        return None
    stripped = text.strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = _OBJECT_RE.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


# Patterns for chain-of-thought answer extraction, in priority order.
# Within each pattern the LAST match wins — LLMs typically state the
# conclusion at the END of their reasoning ("Therefore the answer is B.")
_LETTER_PATTERNS = [
    # LaTeX-boxed answer: \boxed{B}, $\boxed{B}$, \boxed{ b }
    re.compile(r"\\boxed\s*\{\s*([A-Za-z])\s*\}"),
    # "the final answer is B", "correct answer is (B):"
    re.compile(
        r"(?:final|correct|right|best)\s+answer\s+is[:\s]*\(?([A-Za-z])\)?\b",
        re.IGNORECASE,
    ),
    # bare "the answer is B" (must be after the more specific pattern above)
    re.compile(r"\banswer\s+is[:\s]*\(?([A-Za-z])\)?\b", re.IGNORECASE),
    # "answer: B", "Answer = B"
    re.compile(r"\banswer\s*[:=]\s*\(?([A-Za-z])\)?\b", re.IGNORECASE),
    # "Option B is the most appropriate" — verb constraint avoids false
    # positives from per-option analysis sections.
    re.compile(
        r"\b(?:Option|Choice|Selection)\s+([A-Za-z])\b\s+(?:is|would|appears|"
        r"seems|represents|provides|gives|remains|emerges|stands)",
        re.IGNORECASE,
    ),
]


def extract_letter_from_text(
    text: str,
    valid_letters: Optional[set] = None,
) -> str:
    """
    Best-effort extract a single answer letter from free-form rationale text.

    Used as a last-ditch fallback when the LLM emits its answer in
    chain-of-thought form ("$\\boxed{B}$", "the final answer is B",
    "Option B is the most appropriate") rather than populating a
    structured JSON ``predicted_letter`` field.

    ``valid_letters`` constrains which letters are acceptable (e.g.
    ``{"A","B","C","D","E"}`` for MedQA-US). When omitted, accepts any
    A-J.

    Returns ``""`` if no high-confidence match is found.
    """
    if not text:
        return ""
    valid = {s.upper() for s in (valid_letters or set("ABCDEFGHIJ"))}
    for pat in _LETTER_PATTERNS:
        matches = pat.findall(text)
        for letter in reversed(matches):
            up = letter.upper()
            if up in valid:
                return up
    return ""
