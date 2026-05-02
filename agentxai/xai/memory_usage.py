"""
Memory-usage attribution.

The Synthesizer reads every specialist memory key, but reading is not
using. This module produces a per-(owner, key) record of *which* values
appear to have actually informed the Synthesizer's rationale.

The heuristic:

    1. For each memory key the agent wrote, take the latest write's value.
    2. Flatten the value into a list of leaf string tokens (numerics and
       short tokens are skipped to avoid spurious matches).
    3. Count how many tokens appear (case-insensitive substring match) in
       the rationale text.
    4. influence_score = matched_tokens / total_tokens, in [0, 1].
    5. used_in_final_answer = influence_score > 0 (any token cited).

It is deliberately simple and dependency-free. False positives are
possible (a memory value like "MI" can collide with the diagnosis token
in an unrelated rationale) but for typical specialist memory — top
condition lists, retrieved evidence text, guideline match dicts — the
match is informative enough to drive the responsibility score in the
right direction without needing real text-attribution machinery.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional, Sequence

from agentxai.data.schemas import MemoryDiff, MemoryUsage


# Tokens shorter than this are dropped — too many false positives.
_MIN_TOKEN_LEN: int = 3

# Default reader: in the current pipeline only the Synthesizer reads
# specialist memory (via `read_specialist_memories`). Callers can override.
_DEFAULT_READERS: tuple = ("synthesizer",)

# Word boundary: letters / digits / apostrophe / hyphen, must start with a
# letter so pure-numeric runs are skipped (those carry no semantic anchor).
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")

# A tiny stoplist so common English glue words don't trivially match every
# rationale. Kept short on purpose — domain-specific terms like "evidence"
# / "patient" are intentionally NOT here, since their presence in a memory
# value is itself a weak but real citation signal.
_STOPWORDS: frozenset = frozenset({
    "the", "and", "for", "with", "from", "are", "was", "were", "this",
    "that", "but", "not", "any", "all", "have", "has", "had", "can",
    "may", "into", "than", "then", "also", "such", "more", "most",
    "less", "very", "you", "your", "our", "its", "their", "there",
    "these", "those", "his", "her", "him", "she", "him",
})


def attribute_memory_usage(
    rationale: str,
    memory_diffs: Iterable[MemoryDiff],
    *,
    owner_agents: Optional[Sequence[str]] = None,
    reader_agents: Sequence[str] = _DEFAULT_READERS,
) -> List[MemoryUsage]:
    """
    Build a `MemoryUsage` record per (owner, key) by scanning `rationale`
    for substring hits against each value's leaf tokens.

    Parameters
    ----------
    rationale       : The Synthesizer's natural-language rationale string.
                      Empty rationale → every record gets influence=0.0.
    memory_diffs    : All memory diffs for the task. Only `operation == "write"`
                      rows are considered, and only the *latest* write per
                      (agent_id, key) survives.
    owner_agents    : If provided, restrict to these agent_ids. Otherwise,
                      include every agent that has at least one write diff.
    reader_agents   : Agents that read this memory. Stored verbatim on each
                      record so the dashboard can surface "read_by".

    Returns
    -------
    One MemoryUsage record per (agent_id, key), sorted by descending
    influence score so the dashboard shows the most-cited entries first.
    """
    latest = _latest_writes(memory_diffs, owner_agents)
    rationale_lower = (rationale or "").lower()
    readers = list(reader_agents) if reader_agents else []

    out: List[MemoryUsage] = []
    for (agent_id, key), value in latest.items():
        score = _influence_score(value, rationale_lower)
        out.append(
            MemoryUsage(
                agent_id=agent_id,
                key=key,
                read_by=list(readers),
                used_in_final_answer=score > 0.0,
                influence_score=round(score, 4),
            )
        )
    out.sort(key=lambda u: (-u.influence_score, u.agent_id, u.key))
    return out


# ---------------------------------------------------------------------------
# Helpers (exported for testing)
# ---------------------------------------------------------------------------

def _latest_writes(
    memory_diffs: Iterable[MemoryDiff],
    owner_agents: Optional[Sequence[str]],
) -> dict:
    """
    Reduce diffs → {(agent_id, key): final_value_after} keeping the latest
    write per key (timestamp-ordered).
    """
    allow = set(owner_agents) if owner_agents is not None else None
    sorted_diffs = sorted(
        (d for d in memory_diffs if d.operation == "write"),
        key=lambda d: d.timestamp,
    )
    out: dict = {}
    for d in sorted_diffs:
        if allow is not None and d.agent_id not in allow:
            continue
        if not d.agent_id or not d.key:
            continue
        out[(d.agent_id, d.key)] = d.value_after
    return out


def extract_value_tokens(value: Any) -> List[str]:
    """
    Flatten a memory value into a list of word-level tokens used for
    rationale matching.

    Recurses into dicts (values only — dict keys are ignored to avoid
    matching schema labels like "first_line") and sequences. Each leaf
    string is split on word boundaries; tokens are de-duplicated
    case-insensitively and filtered by:

        * minimum length (`_MIN_TOKEN_LEN`)
        * a small stoplist (`_STOPWORDS`)
        * no purely-numeric runs (regex requires a leading letter)

    The resulting list is what `_influence_score` substring-tests against
    the rationale.
    """
    raw_strings: List[str] = []
    _walk_value(value, raw_strings)

    out: List[str] = []
    seen: set = set()
    for s in raw_strings:
        for tok in _WORD_RE.findall(s):
            if len(tok) < _MIN_TOKEN_LEN:
                continue
            tl = tok.lower()
            if tl in _STOPWORDS or tl in seen:
                continue
            seen.add(tl)
            out.append(tok)
    return out


def _walk_value(v: Any, out: List[str]) -> None:
    """Collect every leaf STRING value into `out`. Numerics are skipped."""
    if v is None or isinstance(v, bool):
        return
    if isinstance(v, str):
        t = v.strip()
        if t:
            out.append(t)
        return
    if isinstance(v, (int, float)):
        # Pure numerics rarely anchor a meaningful citation.
        return
    if isinstance(v, dict):
        for sub in v.values():
            _walk_value(sub, out)
        return
    if isinstance(v, (list, tuple, set, frozenset)):
        for sub in v:
            _walk_value(sub, out)
        return
    # Fallback: stringify unknown leaf objects.
    t = str(v).strip()
    if t:
        out.append(t)


def _influence_score(value: Any, rationale_lower: str) -> float:
    """
    Fraction of `value`'s word-level tokens that appear (case-insensitive
    substring) in `rationale_lower`. Returns 0.0 for empty values or
    empty rationale.
    """
    if not rationale_lower:
        return 0.0
    tokens = extract_value_tokens(value)
    if not tokens:
        return 0.0
    matched = sum(1 for tok in tokens if tok.lower() in rationale_lower)
    return matched / len(tokens)
