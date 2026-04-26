"""
Default LLM factory shared by ``agents/base.py`` and the XAI loggers.

Reads one or more Gemini API keys from the environment and returns either a
single ``ChatGoogleGenerativeAI`` (when only one key is configured) or a
``RotatingGeminiLLM`` (when several keys are configured) that round-robins
across keys on rate-limit (429) errors.

Env vars (in priority order):

    GOOGLE_API_KEYS   comma-separated list of keys, e.g. "key1,key2,key3"
    GOOGLE_API_KEY    single key (used if GOOGLE_API_KEYS is unset)
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Key parsing
# ---------------------------------------------------------------------------

def read_gemini_keys() -> List[str]:
    """Return all Gemini API keys configured in the environment."""
    plural = os.environ.get("GOOGLE_API_KEYS", "").strip()
    if plural:
        keys = [k.strip() for k in plural.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("GOOGLE_API_KEY", "").strip()
    if single:
        return [single]
    return []


# ---------------------------------------------------------------------------
# Rate-limit detection
# ---------------------------------------------------------------------------

_RATE_LIMIT_TYPES = {
    "ResourceExhausted",
    "RateLimitError",
    "TooManyRequests",
}
_RATE_LIMIT_MARKERS = ("429", "rate limit", "rate_limit", "quota", "exhausted")


def _is_rate_limit_error(exc: BaseException) -> bool:
    if type(exc).__name__ in _RATE_LIMIT_TYPES:
        return True
    msg = str(exc).lower()
    return any(m in msg for m in _RATE_LIMIT_MARKERS)


# ---------------------------------------------------------------------------
# Rotating wrapper
# ---------------------------------------------------------------------------

class RotatingGeminiLLM:
    """
    Round-robin facade over N ChatGoogleGenerativeAI clients (one per key).

    On invoke(), tries the current key. If the call raises a 429-ish error,
    rotates to the next key and retries. Re-raises the last error after all
    keys have been tried in one full pass — no infinite retries.

    On success, advances the rotation pointer so subsequent calls spread load
    across keys. Pointer state lives on the instance, so all agents that share
    one RotatingGeminiLLM share the same rotation.
    """

    def __init__(self, llms: Sequence[Any]) -> None:
        if not llms:
            raise ValueError("RotatingGeminiLLM requires at least one LLM")
        self._llms: List[Any] = list(llms)
        self._current: int = 0

    def __len__(self) -> int:
        return len(self._llms)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        n = len(self._llms)
        last_exc: Optional[BaseException] = None
        for offset in range(n):
            idx = (self._current + offset) % n
            try:
                result = self._llms[idx].invoke(*args, **kwargs)
                self._current = (idx + 1) % n
                return result
            except Exception as exc:
                if not _is_rate_limit_error(exc):
                    raise
                last_exc = exc
                continue
        assert last_exc is not None
        raise last_exc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SAFETY_BLOCK_NONE = None  # populated lazily once langchain-google-genai is imported


def _build_safety_settings() -> Optional[dict]:
    """Build a {category: BLOCK_NONE} dict, importing the SDK on demand."""
    global _SAFETY_BLOCK_NONE
    if _SAFETY_BLOCK_NONE is not None:
        return _SAFETY_BLOCK_NONE
    try:
        from langchain_google_genai import HarmBlockThreshold, HarmCategory
    except ImportError:
        return None
    _SAFETY_BLOCK_NONE = {
        HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    return _SAFETY_BLOCK_NONE


def build_gemini_llm(model: str, temperature: float = 0.0) -> Any:
    """
    Construct the default LLM for the project. Returns:

      * ``None`` if langchain-google-genai is missing or no key is configured,
        or if every per-key construction raises.
      * A single ``ChatGoogleGenerativeAI`` if exactly one key is configured.
      * A ``RotatingGeminiLLM`` wrapping per-key clients if multiple keys are
        configured.

    Safety filters are forced to BLOCK_NONE — MedQA cases routinely contain
    drug names, overdose scenarios and other content Gemini's defaults flag,
    which silently returns empty content and breaks the JSON-contract prompts.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        return None

    keys = read_gemini_keys()
    if not keys:
        return None

    safety = _build_safety_settings()

    clients: List[Any] = []
    for key in keys:
        try:
            clients.append(
                ChatGoogleGenerativeAI(
                    model=model,
                    temperature=temperature,
                    google_api_key=key,
                    safety_settings=safety,
                )
            )
        except Exception:
            continue

    if not clients:
        return None
    if len(clients) == 1:
        return clients[0]
    return RotatingGeminiLLM(clients)
