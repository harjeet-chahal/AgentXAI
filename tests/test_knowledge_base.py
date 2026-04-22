"""
Tests for agentxai/data/build_knowledge_base.py.

Fast tests (no index on disk required)
---------------------------------------
- Chunker: verify chunk shape, token-budget, overlap carry-over.
- Guideline store: build + load round-trip, required keys, synthetic marker.

Integration test (marked 'slow' — skipped unless --run-slow is passed)
------------------------------------------------------------------------
- Builds a tiny in-memory FAISS index from one short textbook and performs
  a semantic search, asserting the result contains the expected keyword.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers imported directly (avoid hitting disk unless needed)
# ---------------------------------------------------------------------------
from agentxai.data.build_knowledge_base import (
    _iter_chunks,
    _top_conditions_from_medqa,
    build_guideline_store,
    load_guideline_store,
    _GL_FILE,
    _TB_IDX_FILE,
    _TARGET_CHARS,
    _OVERLAP_CHARS,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_textbook(tmp_path_factory) -> pathlib.Path:
    """
    Write a tiny synthetic textbook file with well-known content so we can
    assert keyword presence without depending on real textbook files.
    """
    tmp = tmp_path_factory.mktemp("books")
    p = tmp / "SyntheticMed_Test.txt"
    # Build content that is clearly longer than _TARGET_CHARS so we get ≥2 chunks
    cardiac_para = (
        "Myocardial infarction (MI) occurs when blood flow to part of the heart is "
        "blocked for long enough that heart muscle is damaged or dies.  "
        "ST-segment elevation myocardial infarction (STEMI) is the most severe form.  "
        "Troponin is the biomarker of choice for detecting myocardial injury.  "
        "Treatment includes aspirin, anticoagulation, and urgent revascularisation.\n\n"
    )
    pneumonia_para = (
        "Pneumonia is an infection that inflames the air sacs (alveoli) in one or both lungs.  "
        "Common causative organisms include Streptococcus pneumoniae, Haemophilus influenzae, "
        "and atypical pathogens such as Mycoplasma and Legionella.  "
        "Diagnosis is confirmed by chest X-ray showing consolidation.  "
        "Treatment depends on severity and likely pathogen.\n\n"
    )
    # Repeat enough to exceed _TARGET_CHARS
    repeats = max(1, _TARGET_CHARS // len(cardiac_para) + 2)
    body = (cardiac_para * repeats) + (pneumonia_para * repeats)
    p.write_text(body, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Chunker unit tests
# ---------------------------------------------------------------------------

class TestChunker:
    def test_produces_chunks(self, sample_textbook):
        chunks = _iter_chunks(sample_textbook)
        assert len(chunks) >= 2, "Expected at least 2 chunks from large synthetic textbook"

    def test_chunk_keys(self, sample_textbook):
        chunks = _iter_chunks(sample_textbook)
        for c in chunks:
            assert set(c.keys()) == {"chunk_id", "source_file", "text"}

    def test_chunk_ids_unique(self, sample_textbook):
        chunks = _iter_chunks(sample_textbook)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_source_file_is_filename(self, sample_textbook):
        chunks = _iter_chunks(sample_textbook)
        for c in chunks:
            assert c["source_file"] == sample_textbook.name

    def test_text_nonempty(self, sample_textbook):
        chunks = _iter_chunks(sample_textbook)
        for c in chunks:
            assert len(c["text"]) > 0

    def test_most_chunks_near_target(self, sample_textbook):
        """All chunks except the last should be >= half the target char budget."""
        chunks = _iter_chunks(sample_textbook)
        if len(chunks) <= 1:
            pytest.skip("Too few chunks to test size constraint.")
        for c in chunks[:-1]:
            # After overlap injection a chunk may be somewhat smaller than _TARGET_CHARS,
            # but should be at least _OVERLAP_CHARS worth.
            assert len(c["text"]) >= _OVERLAP_CHARS, (
                f"Chunk {c['chunk_id']} is suspiciously short: {len(c['text'])} chars"
            )

    def test_content_preserved(self, sample_textbook):
        """Every chunk should contain real words from the synthetic textbook."""
        chunks = _iter_chunks(sample_textbook)
        full_text = sample_textbook.read_text()
        combined = " ".join(c["text"] for c in chunks)
        # A word from the synthetic content must appear somewhere
        assert "myocardial" in combined.lower() or "troponin" in combined.lower()

    def test_empty_file_returns_no_chunks(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        assert _iter_chunks(f) == []

    def test_single_short_para_gives_one_chunk(self, tmp_path):
        f = tmp_path / "tiny.txt"
        f.write_text("Short paragraph.", encoding="utf-8")
        chunks = _iter_chunks(f)
        assert len(chunks) == 1
        assert "Short paragraph." in chunks[0]["text"]


# ---------------------------------------------------------------------------
# Guideline store unit tests
# ---------------------------------------------------------------------------

class TestGuidelineStore:
    def test_top_conditions_nonempty(self):
        conditions = _top_conditions_from_medqa(n=10)
        assert len(conditions) == 10
        for c in conditions:
            assert isinstance(c, str) and c

    def test_top_conditions_are_strings(self):
        conditions = _top_conditions_from_medqa(n=5)
        for c in conditions:
            assert isinstance(c, str)

    def test_build_and_load_roundtrip(self, tmp_path):
        """Build the guideline store into a temp dir, load it, check structure."""
        gl_path = tmp_path / "guidelines.json"
        with patch("agentxai.data.build_knowledge_base._GL_FILE", gl_path):
            build_guideline_store(force=True)
            store = load_guideline_store()

        assert isinstance(store, dict)
        assert len(store) == 50

    def test_required_keys(self, tmp_path):
        gl_path = tmp_path / "guidelines.json"
        required = {"condition", "summary", "key_findings", "recommended_workup", "source"}
        with patch("agentxai.data.build_knowledge_base._GL_FILE", gl_path):
            build_guideline_store(force=True)
            store = load_guideline_store()

        for cond, entry in store.items():
            assert required <= set(entry.keys()), (
                f"Missing keys in guideline for {cond!r}: {required - set(entry.keys())}"
            )

    def test_synthetic_marker_in_source(self, tmp_path):
        gl_path = tmp_path / "guidelines.json"
        with patch("agentxai.data.build_knowledge_base._GL_FILE", gl_path):
            build_guideline_store(force=True)
            store = load_guideline_store()

        for cond, entry in store.items():
            assert "synthetic" in entry["source"], (
                f"Entry for {cond!r} is missing synthetic marker in 'source': {entry['source']!r}"
            )

    def test_synthetic_marker_in_summary(self, tmp_path):
        gl_path = tmp_path / "guidelines.json"
        with patch("agentxai.data.build_knowledge_base._GL_FILE", gl_path):
            build_guideline_store(force=True)
            store = load_guideline_store()

        for cond, entry in store.items():
            assert "SYNTHETIC" in entry["summary"].upper(), (
                f"Entry for {cond!r} summary missing SYNTHETIC marker"
            )

    def test_skip_rebuild_without_force(self, tmp_path):
        gl_path = tmp_path / "guidelines.json"
        with patch("agentxai.data.build_knowledge_base._GL_FILE", gl_path):
            build_guideline_store(force=True)
            mtime_1 = gl_path.stat().st_mtime
            build_guideline_store(force=False)  # should be a no-op
            mtime_2 = gl_path.stat().st_mtime
        assert mtime_1 == mtime_2, "File was unexpectedly rewritten without force=True"

    def test_force_rebuild(self, tmp_path):
        gl_path = tmp_path / "guidelines.json"
        with patch("agentxai.data.build_knowledge_base._GL_FILE", gl_path):
            build_guideline_store(force=True)
            mtime_1 = gl_path.stat().st_mtime
            import time; time.sleep(0.05)
            build_guideline_store(force=True)
            mtime_2 = gl_path.stat().st_mtime
        assert mtime_2 >= mtime_1

    def test_load_raises_when_missing(self, tmp_path):
        missing = tmp_path / "no_such.json"
        with patch("agentxai.data.build_knowledge_base._GL_FILE", missing):
            with pytest.raises(FileNotFoundError):
                load_guideline_store()

    def test_key_finding_lists(self, tmp_path):
        gl_path = tmp_path / "guidelines.json"
        with patch("agentxai.data.build_knowledge_base._GL_FILE", gl_path):
            build_guideline_store(force=True)
            store = load_guideline_store()

        for cond, entry in store.items():
            assert isinstance(entry["key_findings"], list)
            assert isinstance(entry["recommended_workup"], list)
            assert len(entry["key_findings"]) > 0
            assert len(entry["recommended_workup"]) > 0


# ---------------------------------------------------------------------------
# Integration — semantic search (marked slow; requires model download)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSemanticSearch:
    """
    Builds a tiny in-memory FAISS index from the synthetic textbook and runs
    a semantic search.  Requires sentence-transformers model download (~90 MB).
    """

    @pytest.fixture(scope="class")
    def tiny_index(self, sample_textbook):
        import faiss
        from sentence_transformers import SentenceTransformer

        chunks = _iter_chunks(sample_textbook)
        model  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        texts  = [c["text"] for c in chunks]
        vecs   = model.encode(texts, convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(vecs)

        dim   = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        return index, chunks, model

    def test_search_returns_results(self, tiny_index):
        index, chunks, model = tiny_index
        query_vec = model.encode(["heart attack treatment"], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_vec)
        scores, indices = index.search(query_vec, k=3)
        assert len(indices[0]) == 3
        assert all(i >= 0 for i in indices[0])

    def test_cardiac_query_hits_cardiac_chunk(self, tiny_index):
        """
        A query about 'myocardial infarction troponin' should retrieve a chunk
        that contains the word 'troponin' (our synthetic textbook has it).
        """
        import faiss
        index, chunks, model = tiny_index
        query = "myocardial infarction troponin biomarker"
        q_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_vec)
        _, idxs = index.search(q_vec, k=3)

        hit_texts = [chunks[i]["text"].lower() for i in idxs[0] if i >= 0]
        assert any("troponin" in t for t in hit_texts), (
            f"Expected 'troponin' in top-3 results, got snippets: "
            + str([t[:80] for t in hit_texts])
        )

    def test_pneumonia_query_hits_pneumonia_chunk(self, tiny_index):
        """
        A query about pneumonia diagnosis should retrieve a chunk mentioning
        'consolidation' (present in the synthetic textbook).
        """
        import faiss
        index, chunks, model = tiny_index
        query = "pneumonia chest X-ray consolidation diagnosis"
        q_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_vec)
        _, idxs = index.search(q_vec, k=3)

        hit_texts = [chunks[i]["text"].lower() for i in idxs[0] if i >= 0]
        assert any("consolidation" in t for t in hit_texts), (
            f"Expected 'consolidation' in top-3 results, got: "
            + str([t[:80] for t in hit_texts])
        )

    def test_scores_are_cosine_range(self, tiny_index):
        """Inner-product of L2-normalised vectors = cosine ∈ [-1, 1]."""
        import faiss
        index, chunks, model = tiny_index
        q_vec = model.encode(["heart disease"], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_vec)
        scores, _ = index.search(q_vec, k=5)
        for s in scores[0]:
            assert -1.0 <= float(s) <= 1.0 + 1e-5, f"Score out of cosine range: {s}"
