"""
Evidence attribution heuristics.

The Synthesizer is asked to emit ``supporting_evidence_ids`` — doc_ids
from Specialist B's ``top_evidence`` that drove its rationale. When the
LLM forgets or returns an empty list, this module infers the same
attribution from the rationale text + retrieved snippets.

It also ranks Specialist B's full ``top_evidence`` by how supportive
each doc is of the final answer — combining whether it was actually
cited with its FAISS retrieval score — for the accountability report's
``most_supportive_evidence_ids`` field.

Design notes:
  * Pure functions over plain dicts/lists. No store, no LLM, no I/O.
  * Word-level substring matching: doc_id in rationale → cited; or any
    significant token from the snippet in rationale → cited.
  * Tokens shorter than 5 characters are dropped — they cause too many
    false positives on common English glue words.
  * The heuristic is deliberately conservative: better to miss a real
    citation than to manufacture a fake one. Reviewers can always look
    at the full top_evidence list to spot un-attributed support.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


# Tokens shorter than this don't reliably anchor a citation.
_MIN_TOKEN_LEN: int = 5
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")

# Cap on tokens scanned per snippet — long passages otherwise produce
# an O(N*M) sweep that adds latency for no signal gain.
_MAX_TOKENS_PER_SNIPPET: int = 30


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_supporting_evidence_ids(
    rationale: str,
    top_evidence: Iterable[Dict[str, Any]],
) -> List[str]:
    """
    Return the doc_ids whose content shows up in `rationale`.

    For each evidence dict (with at least ``doc_id`` and ideally
    ``snippet``/``text``), the doc is considered "cited" when:

      * its ``doc_id`` appears verbatim in the rationale, OR
      * any snippet token of length >= `_MIN_TOKEN_LEN` (case-insensitive)
        appears in the rationale.

    Order matches the input iteration order; duplicates are dropped.
    Returns ``[]`` when either input is empty.
    """
    if not rationale or not top_evidence:
        return []
    rationale_lower = rationale.lower()

    out: List[str] = []
    seen: set = set()
    for ev in top_evidence:
        if not isinstance(ev, dict):
            continue
        doc_id = str(ev.get("doc_id") or "").strip()
        if not doc_id or doc_id in seen:
            continue

        if doc_id.lower() in rationale_lower:
            out.append(doc_id)
            seen.add(doc_id)
            continue

        snippet = str(ev.get("snippet") or ev.get("text") or "")
        tokens = _TOKEN_RE.findall(snippet)
        for tok in tokens[:_MAX_TOKENS_PER_SNIPPET]:
            if len(tok) < _MIN_TOKEN_LEN:
                continue
            if tok.lower() in rationale_lower:
                out.append(doc_id)
                seen.add(doc_id)
                break
    return out


def rank_most_supportive_evidence(
    top_evidence: Iterable[Dict[str, Any]],
    used_ids: Optional[Iterable[str]] = None,
    *,
    limit: int = 5,
) -> List[str]:
    """
    Rank evidence ids by how supportive they are of the final answer.

    Score per doc::

        rank_score = (1.0 if doc_id in used_ids else 0.0)
                   + clamp(retrieval_score, 0, 1)

    So a cited doc with low retrieval score still beats an uncited doc
    with high retrieval score. Stable tie-break by original order.

    Returns the top ``limit`` doc_ids (or all of them if fewer exist).
    Empty/missing inputs yield ``[]``.
    """
    used_set = {str(x) for x in (used_ids or []) if x}
    rows: List[tuple] = []
    for idx, ev in enumerate(top_evidence or []):
        if not isinstance(ev, dict):
            continue
        doc_id = str(ev.get("doc_id") or "").strip()
        if not doc_id:
            continue
        try:
            score = float(ev.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        rank = (1.0 if doc_id in used_set else 0.0) + score
        rows.append((rank, idx, doc_id))

    rows.sort(key=lambda t: (-t[0], t[1]))
    seen: set = set()
    out: List[str] = []
    for _rank, _idx, doc_id in rows:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
        if len(out) >= limit:
            break
    return out


def latest_top_evidence_from_memory_diffs(
    memory_diffs: Iterable[Any],
    *,
    agent_id: str = "specialist_b",
    key: str = "top_evidence",
) -> List[Dict[str, Any]]:
    """
    Reconstruct Specialist B's final ``top_evidence`` value from a list
    of MemoryDiff dicts (or dataclasses with a ``to_dict`` method).

    Used by the accountability report to recover the evidence list
    without needing the live LoggedMemory dict — the report runs after
    the agents have finished, so the live dict may be torn down already.
    """
    latest_value: Optional[List[Dict[str, Any]]] = None
    latest_ts: float = float("-inf")
    for d in memory_diffs or []:
        diff = _to_diff_dict(d)
        if diff.get("agent_id") != agent_id:
            continue
        if diff.get("operation") != "write":
            continue
        if diff.get("key") != key:
            continue
        try:
            ts = float(diff.get("timestamp") or 0.0)
        except (TypeError, ValueError):
            continue
        if ts >= latest_ts:
            value = diff.get("value_after")
            if isinstance(value, list):
                latest_value = value
                latest_ts = ts
    return latest_value or []


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _to_diff_dict(d: Any) -> Dict[str, Any]:
    """Coerce a MemoryDiff dataclass or dict into a plain attribute lookup."""
    if isinstance(d, dict):
        return d
    if hasattr(d, "to_dict"):
        try:
            out = d.to_dict()
            if isinstance(out, dict):
                return out
        except Exception:
            pass
    # Last resort: read attributes directly.
    return {
        "agent_id":  getattr(d, "agent_id", ""),
        "operation": getattr(d, "operation", ""),
        "key":       getattr(d, "key", ""),
        "timestamp": getattr(d, "timestamp", 0.0),
        "value_after": getattr(d, "value_after", None),
    }
