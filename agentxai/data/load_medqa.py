"""
Loader for the local MedQA US dataset (USMLE-style, 5-option English questions).

Expected layout (relative to project root):
    Med_QA/questions/US/train.jsonl
    Med_QA/questions/US/dev.jsonl
    Med_QA/questions/US/test.jsonl
    Med_QA/questions/US/US_qbank.jsonl

Raw record shape:
    {
        "question":   str,          # stem
        "answer":     str,          # full text of the correct option
        "options":    {"A": ..., "E": ...},
        "meta_info":  str,          # e.g. "step1", "step2&3"
        "answer_idx": str,          # letter of the correct option ("A"–"E")
    }

Normalised record shape (what this module returns):
    {
        "task_id":    str,          # zero-padded, e.g. "T00001"
        "question":   str,
        "options":    {"A": ..., "E": ...},
        "answer":     str,          # correct letter, e.g. "E"
        "answer_idx": int,          # 0-based index (A=0 … E=4)
        "meta_info":  str,
        "raw":        dict,         # original record verbatim
    }
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = pathlib.Path(__file__).resolve().parents[2]  # project root
_US_DIR = _ROOT / "Med_QA" / "questions" / "US"

_SPLIT_FILES: Dict[str, pathlib.Path] = {
    "train":  _US_DIR / "train.jsonl",
    "dev":    _US_DIR / "dev.jsonl",
    "test":   _US_DIR / "test.jsonl",
    "qbank":  _US_DIR / "US_qbank.jsonl",
}

_VALID_SPLITS = set(_SPLIT_FILES)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _letter_to_idx(letter: str) -> int:
    """Convert an option letter to a 0-based integer index ("A"→0, "E"→4)."""
    return ord(letter.upper()) - ord("A")


def _normalise(raw: dict, task_id: str) -> Dict:
    """Return a normalised record from one raw JSONL line."""
    letter = raw["answer_idx"].upper()
    return {
        "task_id":    task_id,
        "question":   raw["question"],
        "options":    raw["options"],
        "answer":     letter,
        "answer_idx": _letter_to_idx(letter),
        "meta_info":  raw.get("meta_info", ""),
        "raw":        raw,
    }


def _load_jsonl(path: pathlib.Path) -> List[dict]:
    """Stream-read a JSONL file and return a list of dicts."""
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_medqa_us(split: str = "train") -> List[Dict]:
    """
    Load one split of the US MedQA dataset and return normalised records.

    Parameters
    ----------
    split : {"train", "dev", "test", "qbank"}
        Which file to load.  "qbank" is the combined corpus before splitting.

    Returns
    -------
    List of normalised record dicts (see module docstring for schema).
    """
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(_VALID_SPLITS)}, got {split!r}")

    path = _SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"Expected MedQA file not found: {path}")

    raw_records = _load_jsonl(path)
    prefix = split[0].upper()  # T / D / Te / Q → use first char for readability
    return [
        _normalise(r, task_id=f"{prefix}{i + 1:05d}")
        for i, r in enumerate(raw_records)
    ]


def load_medqa_us_all() -> List[Dict]:
    """
    Concatenate train + dev + test (canonical US splits) and return them with
    globally unique task_ids prefixed "A" (All).

    This is the input to make_splits().
    """
    combined: List[Dict] = []
    for split in ("train", "dev", "test"):
        for rec in load_medqa_us(split):
            combined.append(rec)

    # Re-assign globally unique task_ids
    for i, rec in enumerate(combined):
        rec["task_id"] = f"A{i + 1:05d}"
    return combined


def make_splits(
    records: List[Dict],
    eval_size: int = 1500,
    review_size: int = 500,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Partition *records* into three non-overlapping sets matching the spec:

    - demo   : 10 000 records for trajectory generation / demonstration
    - eval   : 1 500 records for accuracy evaluation (ground-truth answer used)
    - review :   500 records manually reviewed for XAI quality assessment

    Total required = eval_size + review_size + demo (remainder).  If len(records)
    is less than eval_size + review_size the function raises ValueError.

    Parameters
    ----------
    records     : flat list of normalised records (typically from load_medqa_us_all)
    eval_size   : number of records reserved for accuracy evaluation
    review_size : number of records reserved for XAI quality review
    seed        : random seed for reproducible shuffling

    Returns
    -------
    (demo, eval_split, review)  — three disjoint lists
    """
    need = eval_size + review_size
    if len(records) < need:
        raise ValueError(
            f"Not enough records ({len(records)}) for eval ({eval_size}) + "
            f"review ({review_size}) = {need}"
        )

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    review     = shuffled[:review_size]
    eval_split = shuffled[review_size : review_size + eval_size]
    demo       = shuffled[review_size + eval_size :]

    return demo, eval_split, review


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Load and normalise the US MedQA dataset, optionally writing to disk."
    )
    p.add_argument(
        "--split",
        choices=sorted(_VALID_SPLITS),
        default="train",
        help="Which raw split to load (default: train).",
    )
    p.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="Path to write normalised JSONL output (omit to print stats only).",
    )
    p.add_argument(
        "--all-splits",
        action="store_true",
        help=(
            "Load train+dev+test combined, run make_splits(), and write "
            "demo / eval / review to <out-dir>/{demo,eval,review}.jsonl."
        ),
    )
    return p


def _write_jsonl(records: List[Dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records):,} records → {path}")


def main() -> None:
    args = _build_parser().parse_args()

    if args.all_splits:
        print("Loading train + dev + test …")
        all_records = load_medqa_us_all()
        print(f"  total records : {len(all_records):,}")

        demo, eval_split, review = make_splits(all_records)
        print(f"  demo          : {len(demo):,}")
        print(f"  eval          : {len(eval_split):,}")
        print(f"  review        : {len(review):,}")

        if args.out:
            out_dir = args.out
            _write_jsonl(demo,       out_dir / "demo.jsonl")
            _write_jsonl(eval_split, out_dir / "eval.jsonl")
            _write_jsonl(review,     out_dir / "review.jsonl")
    else:
        print(f"Loading split={args.split!r} …")
        records = load_medqa_us(args.split)
        print(f"  loaded {len(records):,} records")

        if args.out:
            _write_jsonl(records, args.out)


if __name__ == "__main__":
    main()
