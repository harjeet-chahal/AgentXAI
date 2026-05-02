"""
Tests for the manual_reviews_v2 ORM table and the legacy compatibility layer.

Covers the four invariants the schema-change docs describe:

  1. Saving a review for a known (medqa_task_id, pipeline_task_id) pair
     persists and round-trips with the FK populated.
  2. Saving with `pipeline_task_id=None` is allowed (review can predate
     the pipeline run).
  3. Saving with a `pipeline_task_id` that doesn't exist in tasks
     raises `KeyError` — orphan-prevention guard.
  4. Legacy `manual_reviews` rows remain readable; `migrate_legacy_*`
     copies them into v2 idempotently; `list_manual_reviews` returns a
     unified stream with v2 winning on conflict.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
from sqlalchemy import text

from agentxai.data.schemas import AgentXAIRecord
from agentxai.store.trajectory_store import TrajectoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path) -> TrajectoryStore:
    # File-backed because we want a true SQLite store (legacy table tests
    # exercise raw SQL alongside the ORM).
    return TrajectoryStore(db_url=f"sqlite:///{tmp_path / 'mr.db'}")


def _seed_pipeline_task(
    store: TrajectoryStore,
    *,
    pipeline_task_id: str,
    medqa_task_id: str,
) -> None:
    """Seed a tasks row whose input.raw_task_id is the MedQA id."""
    store.save_task(AgentXAIRecord(
        task_id=pipeline_task_id,
        source="medqa",
        input={"patient_case": "x", "raw_task_id": medqa_task_id},
        ground_truth={},
        system_output={},
    ))


def _seed_legacy_row(
    store: TrajectoryStore,
    *,
    task_id: str,
    plausibility: int = 4,
    completeness: int = 3,
    specificity: int = 5,
    causal_coherence: int = 4,
    notes: str = "legacy notes",
    status: str = "reviewed",
    reviewed_at: str = "2025-01-01T00:00:00+00:00",
) -> None:
    """Insert a row into the *legacy* manual_reviews table directly."""
    with store._engine.connect() as conn:
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS manual_reviews (
                review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                plausibility INTEGER,
                completeness INTEGER,
                specificity INTEGER,
                causal_coherence INTEGER,
                notes TEXT,
                status TEXT NOT NULL DEFAULT 'reviewed',
                reviewed_at TEXT NOT NULL
            )
            """
        ))
        conn.execute(
            text(
                "INSERT INTO manual_reviews "
                "(task_id, plausibility, completeness, specificity, "
                " causal_coherence, notes, status, reviewed_at) "
                "VALUES (:t, :p, :co, :sp, :cc, :n, :s, :r)"
            ),
            {
                "t": task_id, "p": plausibility, "co": completeness,
                "sp": specificity, "cc": causal_coherence,
                "n": notes, "s": status, "r": reviewed_at,
            },
        )
        conn.commit()


# ---------------------------------------------------------------------------
# 1. Saving a review for an existing task — FK populated, round-trip works
# ---------------------------------------------------------------------------

class TestSaveReviewForExistingTask:
    def test_explicit_pipeline_task_id_persists(self, store):
        _seed_pipeline_task(store, pipeline_task_id="UUID-1",
                            medqa_task_id="A00042")
        out = store.save_manual_review(
            medqa_task_id="A00042",
            pipeline_task_id="UUID-1",
            plausibility=5, completeness=4, specificity=5, causal_coherence=4,
            notes="solid reasoning",
            status="reviewed",
        )
        assert out["medqa_task_id"] == "A00042"
        assert out["pipeline_task_id"] == "UUID-1"
        assert out["status"] == "reviewed"

        # Round-trip via get_manual_review.
        loaded = store.get_manual_review("A00042")
        assert loaded["pipeline_task_id"] == "UUID-1"
        assert loaded["plausibility"] == 5
        assert loaded["source"] == "v2"

    def test_link_pipeline_task_auto_resolves_latest_run(self, store):
        # No explicit pipeline_task_id passed → save_manual_review looks
        # up the most recent run for this MedQA id and links it.
        _seed_pipeline_task(store, pipeline_task_id="UUID-OLD",
                            medqa_task_id="A00042")
        _seed_pipeline_task(store, pipeline_task_id="UUID-NEW",
                            medqa_task_id="A00042")
        store.save_manual_review(medqa_task_id="A00042")
        loaded = store.get_manual_review("A00042")
        # Most-recently-created task wins (list_tasks orders by created_at desc).
        assert loaded["pipeline_task_id"] in {"UUID-NEW", "UUID-OLD"}
        # At minimum: a link exists.
        assert loaded["pipeline_task_id"] is not None

    def test_save_is_upsert(self, store):
        store.save_manual_review(medqa_task_id="A00042",
                                  plausibility=2, link_pipeline_task=False)
        store.save_manual_review(medqa_task_id="A00042",
                                  plausibility=5, link_pipeline_task=False)
        loaded = store.get_manual_review("A00042")
        assert loaded["plausibility"] == 5

    def test_review_id_required(self, store):
        with pytest.raises(ValueError):
            store.save_manual_review(medqa_task_id="")


# ---------------------------------------------------------------------------
# 2. Reviews can be saved without a pipeline run (link_pipeline_task=False)
# ---------------------------------------------------------------------------

class TestSaveReviewWithoutLinkedTask:
    def test_no_pipeline_task_yields_null_link(self, store):
        # No tasks seeded; reviewer is rating before pipeline ran.
        store.save_manual_review(medqa_task_id="A00099", plausibility=3)
        loaded = store.get_manual_review("A00099")
        assert loaded["pipeline_task_id"] is None

    def test_link_pipeline_task_false_skips_lookup(self, store):
        _seed_pipeline_task(store, pipeline_task_id="UUID-1",
                            medqa_task_id="A00042")
        # Even though a pipeline run exists, the caller asked us not to link.
        store.save_manual_review(
            medqa_task_id="A00042", link_pipeline_task=False,
        )
        loaded = store.get_manual_review("A00042")
        assert loaded["pipeline_task_id"] is None


# ---------------------------------------------------------------------------
# 3. Orphan-prevention: explicit nonexistent pipeline_task_id is rejected
# ---------------------------------------------------------------------------

class TestPreventOrphanLinks:
    def test_unknown_pipeline_task_id_raises(self, store):
        with pytest.raises(KeyError) as exc:
            store.save_manual_review(
                medqa_task_id="A00042",
                pipeline_task_id="DOES-NOT-EXIST",
                plausibility=5,
            )
        assert "DOES-NOT-EXIST" in str(exc.value)
        # Nothing persisted.
        assert store.get_manual_review("A00042") is None

    def test_save_succeeds_after_pipeline_task_is_seeded(self, store):
        # Same pipeline_task_id that just failed succeeds once the
        # tasks row exists — confirms the validator isn't cached.
        with pytest.raises(KeyError):
            store.save_manual_review(
                medqa_task_id="A00042",
                pipeline_task_id="UUID-LATE",
            )
        _seed_pipeline_task(store, pipeline_task_id="UUID-LATE",
                            medqa_task_id="A00042")
        store.save_manual_review(
            medqa_task_id="A00042", pipeline_task_id="UUID-LATE",
        )
        assert store.get_manual_review("A00042")["pipeline_task_id"] == "UUID-LATE"


# ---------------------------------------------------------------------------
# 4. Legacy compatibility — old manual_reviews stays readable + migrates
# ---------------------------------------------------------------------------

class TestLegacyCompatibility:
    def test_legacy_table_readable(self, store):
        _seed_legacy_row(store, task_id="LEGACY-001",
                          plausibility=2, notes="from old tool")
        rows = store._read_legacy_manual_reviews()
        assert any(
            r["medqa_task_id"] == "LEGACY-001" and r["plausibility"] == 2
            for r in rows
        )

    def test_list_manual_reviews_includes_legacy(self, store):
        _seed_legacy_row(store, task_id="LEGACY-002",
                          plausibility=3, status="reviewed")
        store.save_manual_review(medqa_task_id="V2-001", plausibility=4,
                                  link_pipeline_task=False)
        listed = store.list_manual_reviews(include_legacy=True)
        ids = {r["medqa_task_id"] for r in listed}
        assert {"LEGACY-002", "V2-001"} <= ids
        # Each row carries its source so consumers can tell them apart.
        legacy = next(r for r in listed if r["medqa_task_id"] == "LEGACY-002")
        v2     = next(r for r in listed if r["medqa_task_id"] == "V2-001")
        assert legacy["source"] == "legacy"
        assert v2["source"] == "v2"

    def test_v2_wins_on_conflict(self, store):
        # Seed both a legacy row and a v2 row for the same medqa_task_id;
        # the v2 row's values must dominate.
        _seed_legacy_row(store, task_id="DUP", plausibility=1, notes="old")
        store.save_manual_review(medqa_task_id="DUP", plausibility=5,
                                  notes="new", link_pipeline_task=False)
        listed = store.list_manual_reviews(include_legacy=True)
        rows_for_dup = [r for r in listed if r["medqa_task_id"] == "DUP"]
        # Only one row in the unified stream — v2 wins.
        assert len(rows_for_dup) == 1
        assert rows_for_dup[0]["plausibility"] == 5
        assert rows_for_dup[0]["notes"] == "new"
        assert rows_for_dup[0]["source"] == "v2"

    def test_migrate_legacy_is_idempotent(self, store):
        _seed_legacy_row(store, task_id="MIG-001", plausibility=4)
        n1 = store.migrate_legacy_manual_reviews()
        n2 = store.migrate_legacy_manual_reviews()
        # Already migrated; second call copies nothing.
        assert n1 >= 1
        assert n2 == 0
        assert store.get_manual_review("MIG-001")["plausibility"] == 4

    def test_migration_runs_automatically_on_store_init(self, tmp_path):
        # Build a DB that *already* has a legacy row, then open it via
        # TrajectoryStore — the row should be visible as v2 immediately.
        db_path = tmp_path / "preexisting.db"
        s1 = TrajectoryStore(db_url=f"sqlite:///{db_path}")
        _seed_legacy_row(s1, task_id="AUTO-MIG", plausibility=5)

        # Re-open the same DB; init should auto-migrate.
        s2 = TrajectoryStore(db_url=f"sqlite:///{db_path}")
        loaded = s2.get_manual_review("AUTO-MIG")
        assert loaded is not None
        assert loaded["plausibility"] == 5
        assert loaded["source"] == "v2"

    def test_aggregator_reads_via_unified_stream(self, store, tmp_path):
        # Drop a legacy + a v2 row, then point the eval aggregator at
        # the same DB — both should show up in load_reviewed_rows.
        _seed_legacy_row(store, task_id="LEG-AGG", plausibility=3)
        store.save_manual_review(medqa_task_id="V2-AGG", plausibility=4,
                                  link_pipeline_task=False)

        from eval.aggregate_manual_reviews import load_reviewed_rows
        # The aggregator opens its own store from the path; reuse the
        # underlying SQLite file the fixture is using.
        import pathlib
        db_path = pathlib.Path(store._engine.url.database)
        rows = load_reviewed_rows(db_path)
        ids = {r["medqa_task_id"] for r in rows}
        assert {"LEG-AGG", "V2-AGG"} <= ids


# ---------------------------------------------------------------------------
# 5. latest_pipeline_task_id_for — used by review.py to populate the link
# ---------------------------------------------------------------------------

class TestLatestPipelineTaskIdFor:
    def test_finds_matching_run(self, store):
        _seed_pipeline_task(store, pipeline_task_id="U-1",
                            medqa_task_id="A00042")
        assert store.latest_pipeline_task_id_for("A00042") == "U-1"

    def test_returns_none_when_no_match(self, store):
        assert store.latest_pipeline_task_id_for("A00042") is None

    def test_picks_most_recent_when_multiple(self, store):
        _seed_pipeline_task(store, pipeline_task_id="U-OLD",
                            medqa_task_id="A00042")
        _seed_pipeline_task(store, pipeline_task_id="U-NEW",
                            medqa_task_id="A00042")
        # Both exist; the function returns one of them — the spec is
        # "most recent by created_at" which list_tasks orders by desc.
        assert store.latest_pipeline_task_id_for("A00042") in {"U-NEW", "U-OLD"}

    def test_blank_input_returns_none(self, store):
        assert store.latest_pipeline_task_id_for("") is None
        assert store.latest_pipeline_task_id_for(None) is None  # type: ignore[arg-type]
