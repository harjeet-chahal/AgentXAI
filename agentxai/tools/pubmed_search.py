"""
``pubmed_search`` tool — local textbook FAISS retrieval (NOT real PubMed).

What this actually does
-----------------------
Despite the name, this tool DOES NOT query the PubMed/NCBI API. It runs a
top-k semantic search over a local FAISS index built from 18 English
medical textbooks (Harrison, Robbins, First Aid, etc.) — see
``agentxai.data.build_knowledge_base.build_textbook_index``. Embeddings
are ``all-MiniLM-L6-v2``; similarity is cosine via L2-normalised inner
product on a ``faiss.IndexFlatIP``.

Why the name stays
------------------
The function name is preserved as a stable integration point: every agent
call site, every stored ``tool_name="pubmed_search"`` row in
``tool_use_events``, and every test that mocks the tool refer to this
identifier. Renaming would break those references for no semantic gain.
The dashboard surfaces this honestly via a display alias
(`pubmed_search (local textbook FAISS)`); see
`agentxai/ui/dashboard.py::_tool_display_name`. To swap in a real PubMed
index later, replace the body of ``pubmed_search`` (or inject a
different ``pubmed_search_fn`` into SpecialistB) without changing any
caller.

Returns
-------
List of ``{doc_id, text, score, source_file}`` dicts, ordered by
descending similarity to the query.
"""

from __future__ import annotations

from typing import List, Optional

import faiss
import numpy as np
from langchain_core.tools import Tool
from sentence_transformers import SentenceTransformer

from agentxai.data.build_knowledge_base import (
    _MODEL_NAME,
    _TB_IDX_FILE,
    build_textbook_index,
    load_textbook_index,
)
from agentxai.data.schemas import AgentXAIRecord
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.tool_provenance import ToolProvenanceLogger, traced_tool


# ---------------------------------------------------------------------------
# Lazy index + model singletons
# ---------------------------------------------------------------------------

_INDEX: Optional[faiss.Index] = None
_METADATA: Optional[List[dict]] = None
_MODEL: Optional[SentenceTransformer] = None


def _ensure_index():
    """Load the textbook FAISS index, building it on first call if absent."""
    global _INDEX, _METADATA
    if _INDEX is None or _METADATA is None:
        if not _TB_IDX_FILE.exists():
            build_textbook_index(force=False)
        _INDEX, _METADATA = load_textbook_index()
    return _INDEX, _METADATA


def _ensure_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(_MODEL_NAME, device="cpu")
    return _MODEL


# ---------------------------------------------------------------------------
# Default in-memory provenance logger (overrideable per agent run)
# ---------------------------------------------------------------------------

_DEFAULT_TASK_ID = "default-pubmed-search-task"
_DEFAULT_STORE   = TrajectoryStore(db_url="sqlite:///:memory:")
_DEFAULT_STORE.save_task(AgentXAIRecord(task_id=_DEFAULT_TASK_ID, source="tool"))
_DEFAULT_LOGGER  = ToolProvenanceLogger(store=_DEFAULT_STORE, task_id=_DEFAULT_TASK_ID)


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------

@traced_tool(_DEFAULT_LOGGER, called_by="default", tool_name="pubmed_search")
def pubmed_search(query: str, k: int = 5) -> List[dict]:
    """
    Retrieve the top-*k* most semantically similar passages to *query*.

    NOTE: This searches a **local FAISS index over 18 medical textbooks**,
    not the PubMed API. The name is preserved for backward compatibility
    with stored tool-call records and call sites; see this module's
    docstring for the full rationale and the swap-in path.

    Returns
    -------
    List of dicts with keys: doc_id, text, score, source_file.
    """
    if not query or not query.strip():
        return []
    if k <= 0:
        return []

    index, metadata = _ensure_index()
    model = _ensure_model()

    vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(vec)

    k_eff = min(k, index.ntotal)
    scores, ids = index.search(vec, k_eff)

    results: List[dict] = []
    for s, i in zip(scores[0].tolist(), ids[0].tolist()):
        if i < 0:
            continue
        meta = metadata[i]
        results.append({
            "doc_id":      meta["chunk_id"],
            "text":        meta["text"],
            "score":       float(s),
            "source_file": meta["source_file"],
        })
    return results


# ---------------------------------------------------------------------------
# LangChain Tool wrapper
# ---------------------------------------------------------------------------

pubmed_search_tool = Tool(
    name="pubmed_search",
    description=(
        "Top-k semantic search over a LOCAL medical-textbook FAISS index "
        "(18 English textbooks, all-MiniLM-L6-v2 embeddings). Despite the "
        "name, this is NOT the PubMed/NCBI API — the name is preserved as "
        "a stable integration point for legacy call sites and stored "
        "records. Returns: list of {doc_id, text, score, source_file}."
    ),
    func=pubmed_search,
)
