"""
guideline_lookup tool — fuzzy-match a free-text condition name against the
local synthetic guideline store and return its (stub) guideline record.

The store is built by ``agentxai.data.build_knowledge_base.build_guideline_store``
from the top-50 most common MedQA US answer-option strings. Every record is a
SYNTHETIC STUB and is marked as such — not for clinical use.

Match strategy: rapidfuzz WRatio against the condition keys, score cutoff 60
(scaled 0–100). Below cutoff returns ``{"match": None}``.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import Tool
from rapidfuzz import fuzz, process

from agentxai.data.build_knowledge_base import (
    _GL_FILE,
    build_guideline_store,
    load_guideline_store,
)
from agentxai.data.schemas import AgentXAIRecord
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.tool_provenance import ToolProvenanceLogger, traced_tool


# ---------------------------------------------------------------------------
# Lazy guideline-store singleton
# ---------------------------------------------------------------------------

_STORE: Optional[dict] = None
_MATCH_CUTOFF: float = 60.0


def _ensure_store() -> dict:
    global _STORE
    if _STORE is None:
        if not _GL_FILE.exists():
            build_guideline_store(force=False)
        _STORE = load_guideline_store()
    return _STORE


# ---------------------------------------------------------------------------
# Default in-memory provenance logger (overrideable per agent run)
# ---------------------------------------------------------------------------

_DEFAULT_TASK_ID = "default-guideline-lookup-task"
_DEFAULT_STORE_DB = TrajectoryStore(db_url="sqlite:///:memory:")
_DEFAULT_STORE_DB.save_task(AgentXAIRecord(task_id=_DEFAULT_TASK_ID, source="tool"))
_DEFAULT_LOGGER  = ToolProvenanceLogger(store=_DEFAULT_STORE_DB, task_id=_DEFAULT_TASK_ID)


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------

@traced_tool(_DEFAULT_LOGGER, called_by="default", tool_name="guideline_lookup")
def guideline_lookup(condition: str) -> dict:
    """
    Fuzzy-match *condition* against the synthetic guideline store and return
    the best record. Returns ``{"match": None}`` if no key scores above the
    cutoff (60 / 100) — the caller should treat this as "no guideline found".

    The returned record (when matched) augments the stored guideline dict with:
      - ``match``       : the matched key (the canonical condition name)
      - ``match_score`` : the similarity score normalised to 0–1
    """
    if not condition or not condition.strip():
        return {"match": None}

    store = _ensure_store()
    keys = list(store.keys())
    if not keys:
        return {"match": None}

    hit = process.extractOne(
        condition.strip(),
        keys,
        scorer=fuzz.WRatio,
        score_cutoff=_MATCH_CUTOFF,
    )
    if hit is None:
        return {"match": None}

    matched_key, score, _ = hit
    record = dict(store[matched_key])
    record["match"] = matched_key
    record["match_score"] = round(score / 100.0, 4)
    return record


# ---------------------------------------------------------------------------
# LangChain Tool wrapper
# ---------------------------------------------------------------------------

guideline_lookup_tool = Tool(
    name="guideline_lookup",
    description=(
        "Fuzzy-match a condition name against the local (synthetic) guideline "
        "store and return its stub guideline record, or {'match': None} if no "
        "match is above the similarity cutoff."
    ),
    func=guideline_lookup,
)
