"""
Re-score recent tasks against the corrected ground-truth comparison without
re-running the pipeline. Reads predicted letters from the DB, looks up the
true letter from the MedQA test split by matching the question text, and
prints the corrected accuracy.

Usage:
    python rescore_last_run.py            # rescore the most recent 100 tasks
    python rescore_last_run.py --limit 50
"""
from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
from sqlalchemy import text

from agentxai.data.load_medqa import load_medqa_us
from agentxai.store.trajectory_store import TrajectoryStore
from run_pipeline import _resolve_correct_letter


def main(limit: int) -> None:
    store = TrajectoryStore()
    with store._engine.connect() as conn:
        rows = conn.execute(text(
            f"SELECT task_id FROM tasks ORDER BY created_at DESC LIMIT {limit}"
        )).fetchall()

    if not rows:
        print("No tasks in DB.")
        return

    # Build a question-text index over both train and test so we can match
    # tasks regardless of which split they came from.
    print(f"Indexing MedQA splits for question-text lookup …")
    medqa_index: dict[str, dict] = {}
    for split in ("train", "dev", "test"):
        try:
            for r in load_medqa_us(split):
                medqa_index[r["question"].strip()] = r
        except FileNotFoundError:
            pass
    print(f"Indexed {len(medqa_index):,} MedQA records.\n")

    n_total = 0
    n_correct = 0
    n_unmatched = 0
    n_no_pred = 0

    print(f"{'task_id':12}  {'pred':>4}  {'truth':>5}  {'correct':>7}")
    print("-" * 48)

    for (tid,) in rows:
        rec = store.get_full_record(tid)
        pred = (rec.system_output.get("predicted_letter") or "").strip().upper()
        case_text = (rec.input.get("patient_case") or "").strip()

        medqa = medqa_index.get(case_text)
        if medqa is None:
            n_unmatched += 1
            continue

        truth = _resolve_correct_letter(medqa)
        if not pred:
            n_no_pred += 1

        is_correct = bool(pred and pred == truth)
        n_total += 1
        n_correct += int(is_correct)

        print(f"{tid[:8]}      {pred:>4}  {truth:>5}  {'yes' if is_correct else 'no':>7}")

    print("-" * 48)
    print()
    print(f"Scored:        {n_total} / {len(rows)} tasks (matched in MedQA index)")
    print(f"Unmatched:     {n_unmatched}  (couldn't find question text — UI submissions or different split)")
    print(f"Empty pred:    {n_no_pred}  (synthesizer never produced a diagnosis letter)")
    if n_total:
        print()
        print(f"Corrected accuracy: {n_correct}/{n_total} = {n_correct / n_total:.1%}")
        # Random baseline for context
        baseline = 1.0 / 5  # 5-option MCQ
        lift = (n_correct / n_total) / baseline
        print(f"Random baseline:    {baseline:.1%}   (lift: {lift:.2f}x)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=100)
    args = p.parse_args()
    main(args.limit)
