"""
Tests for agentxai/data/load_medqa.py.

Loads 10 records from the US train split and asserts the full normalised schema.
Keeps the test fast by never reading more than needed.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from agentxai.data.load_medqa import (
    _letter_to_idx,
    load_medqa_us,
    load_medqa_us_all,
    make_splits,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ten_records():
    """Load exactly 10 records from the US train split (cheap — streams the file)."""
    all_records = load_medqa_us("train")
    return all_records[:10]


# ---------------------------------------------------------------------------
# _letter_to_idx
# ---------------------------------------------------------------------------

class TestLetterToIdx:
    def test_a_is_zero(self):
        assert _letter_to_idx("A") == 0

    def test_e_is_four(self):
        assert _letter_to_idx("E") == 4

    def test_case_insensitive(self):
        assert _letter_to_idx("b") == _letter_to_idx("B") == 1


# ---------------------------------------------------------------------------
# Normalised schema — one record at a time
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"task_id", "question", "options", "answer", "answer_idx", "meta_info", "raw"}


class TestNormalisedSchema:
    def test_has_all_keys(self, ten_records):
        for rec in ten_records:
            assert REQUIRED_KEYS == set(rec.keys()), (
                f"Missing or extra keys in record {rec.get('task_id')}: "
                f"got {set(rec.keys())}"
            )

    def test_task_id_format(self, ten_records):
        for i, rec in enumerate(ten_records):
            assert rec["task_id"] == f"T{i + 1:05d}"

    def test_question_is_nonempty_string(self, ten_records):
        for rec in ten_records:
            assert isinstance(rec["question"], str)
            assert len(rec["question"]) > 0

    def test_options_is_dict_with_letter_keys(self, ten_records):
        for rec in ten_records:
            opts = rec["options"]
            assert isinstance(opts, dict)
            assert len(opts) >= 4, "US set has 4–5 options"
            for key in opts:
                assert key in "ABCDE", f"Unexpected option key: {key!r}"
            for val in opts.values():
                assert isinstance(val, str) and val

    def test_answer_is_letter_in_options(self, ten_records):
        for rec in ten_records:
            assert rec["answer"] in rec["options"], (
                f"answer={rec['answer']!r} not in options keys {list(rec['options'].keys())}"
            )

    def test_answer_idx_is_correct_int(self, ten_records):
        for rec in ten_records:
            expected = ord(rec["answer"]) - ord("A")
            assert rec["answer_idx"] == expected, (
                f"answer={rec['answer']!r} should give answer_idx={expected}, "
                f"got {rec['answer_idx']}"
            )

    def test_answer_idx_in_valid_range(self, ten_records):
        for rec in ten_records:
            assert 0 <= rec["answer_idx"] <= 4

    def test_answer_text_matches_option(self, ten_records):
        """The text at options[answer] must equal raw['answer']."""
        for rec in ten_records:
            letter = rec["answer"]
            assert rec["options"][letter] == rec["raw"]["answer"], (
                f"options[{letter!r}] != raw answer for task {rec['task_id']}"
            )

    def test_meta_info_is_string(self, ten_records):
        for rec in ten_records:
            assert isinstance(rec["meta_info"], str)

    def test_raw_preserves_original_keys(self, ten_records):
        for rec in ten_records:
            raw = rec["raw"]
            assert "question" in raw
            assert "answer" in raw
            assert "options" in raw
            assert "answer_idx" in raw

    def test_json_serialisable(self, ten_records):
        for rec in ten_records:
            json.dumps(rec)  # must not raise


# ---------------------------------------------------------------------------
# load_medqa_us — split validation
# ---------------------------------------------------------------------------

class TestLoadMedqaUs:
    def test_bad_split_raises(self):
        with pytest.raises(ValueError, match="split must be one of"):
            load_medqa_us("bogus")

    def test_train_nonempty(self):
        records = load_medqa_us("train")
        assert len(records) > 0

    def test_dev_nonempty(self):
        assert len(load_medqa_us("dev")) > 0

    def test_test_nonempty(self):
        assert len(load_medqa_us("test")) > 0

    def test_task_ids_unique(self, ten_records):
        ids = [r["task_id"] for r in ten_records]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# load_medqa_us_all
# ---------------------------------------------------------------------------

class TestLoadMedqaUsAll:
    @pytest.fixture(scope="class")
    def all_records(self):
        return load_medqa_us_all()

    def test_count_is_sum_of_splits(self, all_records):
        n_train = len(load_medqa_us("train"))
        n_dev   = len(load_medqa_us("dev"))
        n_test  = len(load_medqa_us("test"))
        assert len(all_records) == n_train + n_dev + n_test

    def test_task_ids_globally_unique(self, all_records):
        ids = [r["task_id"] for r in all_records]
        assert len(ids) == len(set(ids))

    def test_task_ids_use_A_prefix(self, all_records):
        for rec in all_records:
            assert rec["task_id"].startswith("A"), rec["task_id"]


# ---------------------------------------------------------------------------
# make_splits
# ---------------------------------------------------------------------------

class TestMakeSplits:
    @pytest.fixture(scope="class")
    def splits(self):
        records = load_medqa_us_all()
        return records, make_splits(records, eval_size=1500, review_size=500, seed=42)

    def test_sizes(self, splits):
        records, (demo, eval_split, review) = splits
        assert len(review)     == 500
        assert len(eval_split) == 1500
        assert len(demo)       == len(records) - 2000

    def test_disjoint(self, splits):
        _, (demo, eval_split, review) = splits
        demo_ids   = {r["task_id"] for r in demo}
        eval_ids   = {r["task_id"] for r in eval_split}
        review_ids = {r["task_id"] for r in review}
        assert demo_ids.isdisjoint(eval_ids)
        assert demo_ids.isdisjoint(review_ids)
        assert eval_ids.isdisjoint(review_ids)

    def test_union_covers_all(self, splits):
        records, (demo, eval_split, review) = splits
        all_ids = {r["task_id"] for r in records}
        union   = {r["task_id"] for r in demo + eval_split + review}
        assert union == all_ids

    def test_reproducible(self):
        records = load_medqa_us_all()
        demo1, eval1, review1 = make_splits(records, seed=42)
        demo2, eval2, review2 = make_splits(records, seed=42)
        assert [r["task_id"] for r in demo1]   == [r["task_id"] for r in demo2]
        assert [r["task_id"] for r in eval1]   == [r["task_id"] for r in eval2]
        assert [r["task_id"] for r in review1] == [r["task_id"] for r in review2]

    def test_different_seed_gives_different_order(self):
        records = load_medqa_us_all()
        demo1, _, _ = make_splits(records, seed=42)
        demo2, _, _ = make_splits(records, seed=99)
        ids1 = [r["task_id"] for r in demo1]
        ids2 = [r["task_id"] for r in demo2]
        assert ids1 != ids2

    def test_too_few_records_raises(self):
        tiny = [{"task_id": f"X{i}"} for i in range(10)]
        with pytest.raises(ValueError, match="Not enough records"):
            make_splits(tiny, eval_size=8, review_size=5)


# ---------------------------------------------------------------------------
# CLI — write to disk
# ---------------------------------------------------------------------------

class TestCLI:
    def test_single_split_write(self, tmp_path):
        """Run the CLI programmatically and verify the output file."""
        import sys
        from unittest.mock import patch

        out = tmp_path / "medqa_us.jsonl"
        with patch.object(sys, "argv", ["load_medqa", "--split", "train", "--out", str(out)]):
            from agentxai.data.load_medqa import main
            main()

        assert out.exists()
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) > 0
        first = json.loads(lines[0])
        assert set(first.keys()) >= REQUIRED_KEYS

    def test_all_splits_write(self, tmp_path):
        """--all-splits creates demo.jsonl, eval.jsonl, review.jsonl."""
        import sys
        from unittest.mock import patch

        with patch.object(sys, "argv", [
            "load_medqa", "--all-splits", "--out", str(tmp_path)
        ]):
            from agentxai.data.load_medqa import main
            main()

        for name in ("demo.jsonl", "eval.jsonl", "review.jsonl"):
            p = tmp_path / name
            assert p.exists(), f"{name} not written"
            lines = p.read_text(encoding="utf-8").splitlines()
            assert len(lines) > 0
