"""
End-to-end smoke test for ``run_pipeline.Pipeline``.

Runs ONE real MedQA record through the full multi-agent + XAI pipeline
against a real Claude LLM. Marked ``slow`` because it costs a handful of
API calls; pass ``--run-slow`` to enable.

The ``pubmed_search`` tool is stubbed because the local FAISS textbook
index segfaults during build on Apple Silicon (numpy 2.4.4 + torch 2.11);
the rest of the tools (symptom_lookup, severity_scorer, guideline_lookup)
hit their real cached indices.

The test asserts that every XAI table — trajectory, plans, tool_calls,
memory_diffs, messages, causal_edges, accountability_reports — gained at
least one row, plus that the system_output and ground_truth blocks are
populated.
"""

from __future__ import annotations

import os

import pytest

from agentxai.data.load_medqa import load_medqa_us
from agentxai.store.trajectory_store import TrajectoryStore

from run_pipeline import Pipeline


# A minimal substitute corpus so SpecialistB has something to "retrieve" without
# loading the FAISS textbook index.
_FAKE_DOCS = [
    {"doc_id": "Harrison__0001",
     "text":   "Acute coronary syndrome workup includes ECG, troponin, and aspirin.",
     "score":  0.91, "source_file": "InternalMed_Harrison.txt"},
    {"doc_id": "Harrison__0002",
     "text":   "STEMI is treated with primary PCI within 90 minutes of presentation.",
     "score":  0.87, "source_file": "InternalMed_Harrison.txt"},
    {"doc_id": "FA__0050",
     "text":   "First Aid: classic STEMI ECG features include ST elevation in II, III, aVF.",
     "score":  0.82, "source_file": "First_Aid_Step2.txt"},
    {"doc_id": "Pathoma__0010",
     "text":   "Coronary thrombosis on a ruptured atherosclerotic plaque drives MI.",
     "score":  0.76, "source_file": "Pathoma_Husain.txt"},
    {"doc_id": "Robbins__0099",
     "text":   "Myocardial necrosis follows >20 minutes of ischemia with reperfusion.",
     "score":  0.71, "source_file": "Pathology_Robbins.txt"},
]


def _stub_pubmed_search(query: str, k: int = 5):
    return list(_FAKE_DOCS[: max(0, k)])


def _row_count(store: TrajectoryStore, table: str) -> int:
    from sqlalchemy import text
    with store._engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0


@pytest.mark.slow
def test_pipeline_end_to_end_on_one_medqa_record(tmp_path):
    """Real LLM, real cached tool indices, stubbed PubMed (FAISS)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set; skipping live LLM smoke test.")

    db_path = tmp_path / "smoke.db"
    db_url = f"sqlite:///{db_path}"

    pipeline = Pipeline(db_url=db_url, pubmed_search_fn=_stub_pubmed_search)

    record = load_medqa_us("train")[0]
    result = pipeline.run_task(record)

    # ----- Top-level record -----
    assert result.task_id
    assert result.input.get("patient_case")
    assert result.ground_truth.get("correct_answer") in record["options"]
    so = result.system_output
    assert "final_diagnosis" in so
    assert "confidence" in so
    assert "predicted_letter" in so
    assert "correct" in so

    # ----- Every XAI table populated -----
    xai = result.xai_data
    assert len(xai.trajectory) > 0,            "trajectory_events empty"
    assert len(xai.plans) > 0,                 "agent_plans empty"
    assert len(xai.tool_calls) > 0,            "tool_use_events empty"
    assert len(xai.memory_diffs) > 0,          "memory_diffs empty"
    assert len(xai.messages) > 0,              "agent_messages empty"
    assert len(xai.causal_graph.edges) > 0,    "causal_edges empty"
    assert xai.accountability_report is not None, "accountability_report missing"

    # And re-confirm at the SQL level — the assertion above runs through the
    # ORM, so a sanity-check straight against the underlying tables protects
    # against any reader bug masking an empty table.
    store = TrajectoryStore(db_url=db_url)
    for table in (
        "trajectory_events", "agent_plans", "tool_use_events", "memory_diffs",
        "agent_messages", "causal_edges", "accountability_reports",
    ):
        assert _row_count(store, table) > 0, f"{table} has zero rows"
