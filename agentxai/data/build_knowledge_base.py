"""
Builds two local knowledge bases used by Specialist B's tools:

1. Textbook FAISS index
   - Chunks 18 English medical textbooks (Med_QA/textbooks/en/) into ~500-token
     passages (≈2 000 chars) by accumulating blank-line-separated paragraphs.
   - Embeds each chunk with sentence-transformers/all-MiniLM-L6-v2 (CPU-only).
   - Writes:
       data/indices/textbooks/index.faiss      — FAISS IndexFlatIP (inner-product,
                                                 i.e. cosine after L2-norm)
       data/indices/textbooks/metadata.jsonl   — parallel per-chunk metadata

2. Guideline store
   ⚠️  SYNTHETIC STUBS — not real clinical guidelines.
   - Counts the 50 most common conditions in US MedQA answer-option text.
   - For each condition writes a structured stub dict so downstream tools have
     something to retrieve.  Replace with real guideline data before any
     clinical use.
   - Writes:
       data/indices/guidelines.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import re
import sys
import time
from collections import Counter
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT        = pathlib.Path(__file__).resolve().parents[2]
_TB_DIR      = _ROOT / "Med_QA" / "textbooks" / "en"
_US_TRAIN    = _ROOT / "Med_QA" / "questions" / "US" / "train.jsonl"

_IDX_ROOT    = _ROOT / "agentxai" / "data" / "indices"
_TB_IDX_DIR  = _IDX_ROOT / "textbooks"
_TB_IDX_FILE = _TB_IDX_DIR / "index.faiss"
_TB_META_FILE= _TB_IDX_DIR / "metadata.jsonl"
_GL_FILE     = _IDX_ROOT  / "guidelines.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
_TARGET_CHARS    = 2_000   # ≈500 tokens
_OVERLAP_CHARS   = 200     # carry-over between consecutive chunks
_EMBED_BATCH     = 64
_LOG_EVERY       = 500     # chunks between progress messages
_TOP_N_COND      = 50

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1 — Textbook chunker
# ---------------------------------------------------------------------------

def _iter_chunks(path: pathlib.Path) -> List[dict]:
    """
    Split one textbook into ~500-token chunks.

    Strategy: accumulate blank-line-separated paragraphs until the running
    character count crosses _TARGET_CHARS, then emit a chunk.  The last
    paragraph of each chunk is prepended to the next one (_OVERLAP_CHARS
    worth of carry-over) so retrieval at chunk boundaries is robust.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: List[dict] = []
    buf: List[str] = []
    buf_chars = 0
    chunk_idx = 0

    def _emit() -> None:
        nonlocal chunk_idx, buf, buf_chars
        body = "\n\n".join(buf).strip()
        if body:
            chunks.append({
                "chunk_id":    f"{path.stem}__{chunk_idx:04d}",
                "source_file": path.name,
                "text":        body,
            })
            chunk_idx += 1
        # carry-over: keep the last paragraph (up to _OVERLAP_CHARS)
        if buf:
            last = buf[-1]
            if len(last) <= _OVERLAP_CHARS:
                buf = [last]
                buf_chars = len(last)
            else:
                buf = []
                buf_chars = 0
        else:
            buf_chars = 0

    for para in paragraphs:
        buf.append(para)
        buf_chars += len(para)
        if buf_chars >= _TARGET_CHARS:
            _emit()

    if buf_chars > 0:          # flush remainder
        _emit()

    return chunks


def _load_model() -> SentenceTransformer:
    log.info("Loading embedding model %s …", _MODEL_NAME)
    model = SentenceTransformer(_MODEL_NAME, device="cpu")
    return model


# ---------------------------------------------------------------------------
# Public: build textbook index
# ---------------------------------------------------------------------------

def build_textbook_index(force: bool = False) -> None:
    """
    Chunk all English textbooks, embed with all-MiniLM-L6-v2, and write a
    FAISS IndexFlatIP (cosine-equivalent after L2 normalisation) plus a
    parallel metadata JSONL file.

    Parameters
    ----------
    force : if True, rebuild even when index files already exist.
    """
    if not force and _TB_IDX_FILE.exists() and _TB_META_FILE.exists():
        log.info("Textbook index already exists at %s — skipping (pass force=True to rebuild).",
                 _TB_IDX_DIR)
        return

    _TB_IDX_DIR.mkdir(parents=True, exist_ok=True)

    # --- collect chunks ---
    txt_files = sorted(_TB_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under {_TB_DIR}")

    log.info("Chunking %d textbook files …", len(txt_files))
    all_chunks: List[dict] = []
    for f in txt_files:
        chunks = _iter_chunks(f)
        log.info("  %-40s  →  %5d chunks", f.name, len(chunks))
        all_chunks.extend(chunks)
    log.info("Total chunks: %d", len(all_chunks))

    # --- embed ---
    model = _load_model()
    texts = [c["text"] for c in all_chunks]
    dim   = model.get_sentence_embedding_dimension()
    embeddings = np.zeros((len(texts), dim), dtype=np.float32)

    t0 = time.perf_counter()
    for start in range(0, len(texts), _EMBED_BATCH):
        batch = texts[start : start + _EMBED_BATCH]
        vecs  = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings[start : start + len(batch)] = vecs

        done = start + len(batch)
        if done % _LOG_EVERY < _EMBED_BATCH or done == len(texts):
            elapsed = time.perf_counter() - t0
            rate    = done / elapsed if elapsed > 0 else 0
            log.info("  embedded %6d / %d chunks  (%.0f chunks/s)", done, len(texts), rate)

    # L2-normalise so inner-product ≡ cosine similarity
    faiss.normalize_L2(embeddings)

    # --- build FAISS index ---
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(_TB_IDX_FILE))
    log.info("FAISS index written → %s  (%d vectors, dim=%d)", _TB_IDX_FILE, index.ntotal, dim)

    # --- write metadata ---
    with _TB_META_FILE.open("w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    log.info("Metadata written → %s", _TB_META_FILE)


# ---------------------------------------------------------------------------
# Public: load textbook index
# ---------------------------------------------------------------------------

def load_textbook_index() -> Tuple[faiss.Index, List[dict]]:
    """
    Load the pre-built FAISS index and parallel metadata list from disk.

    Returns
    -------
    (index, metadata_list)
        index         : faiss.IndexFlatIP, ready for .search()
        metadata_list : list of dicts with keys chunk_id, source_file, text
                        — index i in the list corresponds to vector i in the index.

    Raises
    ------
    FileNotFoundError if either file is missing (run build_textbook_index first).
    """
    if not _TB_IDX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {_TB_IDX_FILE}. "
            "Run build_textbook_index() or: python -m agentxai.data.build_knowledge_base --target textbooks"
        )
    if not _TB_META_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found at {_TB_META_FILE}.")

    index = faiss.read_index(str(_TB_IDX_FILE))
    metadata: List[dict] = []
    with _TB_META_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))

    if index.ntotal != len(metadata):
        raise RuntimeError(
            f"Index/metadata mismatch: {index.ntotal} vectors vs {len(metadata)} metadata rows."
        )
    return index, metadata


# ---------------------------------------------------------------------------
# 2 — Synthetic guideline store
# ---------------------------------------------------------------------------

_STUB_TEMPLATE = {
    # Filled per-condition; values below are defaults overridden inside the loop.
    "condition": "",
    "summary": (
        "SYNTHETIC STUB — not a real clinical guideline. "
        "Consult authoritative sources (AHA, USPSTF, UpToDate) for clinical use."
    ),
    "key_findings": [],
    "recommended_workup": [],
    "source": "synthetic/medqa-derived",
}

# Generic workup steps to sprinkle into stubs so they're not completely empty.
_GENERIC_WORKUP = [
    "Detailed history and physical examination",
    "Basic metabolic panel (BMP)",
    "Complete blood count (CBC)",
    "ECG if cardiovascular involvement suspected",
    "Relevant imaging per clinical presentation",
    "Specialist referral as indicated",
]


def _top_conditions_from_medqa(n: int = _TOP_N_COND) -> List[str]:
    """Count answer-option text across the US train split; return top-n conditions."""
    counter: Counter = Counter()
    with _US_TRAIN.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            for opt_text in rec.get("options", {}).values():
                # Normalise: strip trailing punctuation, title-case
                cleaned = opt_text.strip().rstrip(".").strip()
                if cleaned:
                    counter[cleaned] += 1
    return [condition for condition, _ in counter.most_common(n)]


def build_guideline_store(force: bool = False) -> None:
    """
    Build a synthetic guideline store from the top-50 MedQA conditions.

    ⚠️  SYNTHETIC STUBS — these are not real clinical guidelines.
    They are generated solely to give Specialist B's guideline_lookup tool
    something to retrieve during development and testing.  Every record
    carries an explicit 'source': 'synthetic/medqa-derived' marker.

    Parameters
    ----------
    force : if True, rebuild even when guidelines.json already exists.
    """
    if not force and _GL_FILE.exists():
        log.info("Guideline store already exists at %s — skipping (pass force=True to rebuild).",
                 _GL_FILE)
        return

    _GL_FILE.parent.mkdir(parents=True, exist_ok=True)

    log.info("Extracting top-%d conditions from MedQA US train …", _TOP_N_COND)
    conditions = _top_conditions_from_medqa(_TOP_N_COND)
    log.info("Top conditions: %s", conditions[:10])

    guidelines = {}
    for cond in conditions:
        guidelines[cond] = {
            "condition":          cond,
            "summary": (
                f"[SYNTHETIC STUB] {cond} is a clinical condition encountered in "
                f"USMLE-style board questions. This stub was auto-generated from "
                f"MedQA answer-option frequency data and is NOT a real guideline."
            ),
            "key_findings": [
                f"History and presentation consistent with {cond}",
                "Physical exam findings vary by severity",
                "Laboratory and imaging findings documented in primary literature",
            ],
            "recommended_workup": _GENERIC_WORKUP,
            "source": "synthetic/medqa-derived",
        }

    with _GL_FILE.open("w", encoding="utf-8") as fh:
        json.dump(guidelines, fh, indent=2, ensure_ascii=False)

    log.info("Guideline store written → %s  (%d conditions)", _GL_FILE, len(guidelines))


def load_guideline_store() -> dict:
    """Return the guideline store as {condition_str: guideline_dict}."""
    if not _GL_FILE.exists():
        raise FileNotFoundError(
            f"Guideline store not found at {_GL_FILE}. "
            "Run build_guideline_store() or: python -m agentxai.data.build_knowledge_base --target guidelines"
        )
    with _GL_FILE.open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build local knowledge bases for AgentXAI Specialist B tools."
    )
    p.add_argument(
        "--target",
        choices=["textbooks", "guidelines", "all"],
        default="all",
        help=(
            "Which index to build: 'textbooks' (FAISS), 'guidelines' (JSON store), "
            "or 'all' (default)."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if index files already exist.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    t0 = time.perf_counter()

    if args.target in ("guidelines", "all"):
        build_guideline_store(force=args.force)

    if args.target in ("textbooks", "all"):
        build_textbook_index(force=args.force)

    log.info("Done in %.1f s.", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
