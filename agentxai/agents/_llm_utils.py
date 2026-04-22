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
