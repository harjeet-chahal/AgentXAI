"""
Re-run ONLY the one-line explanation for a stored task — keeps every
existing analysis (responsibility scores, root cause, attributions) intact
and just regenerates the natural-language sentence through the current
LLM prompt + fallback rules.

Usage:
    python regen_explanation.py <task_id>          # by task id
    python regen_explanation.py --latest           # most recent task in the DB
    python regen_explanation.py --all              # every task in the DB

Persists the new explanation back to the accountability_reports table so
the dashboard sees it on next render.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

# Load .env BEFORE importing anything that might init an LLM client — the
# pipeline / dashboard do this in their own entrypoints, but a standalone
# script has to do it itself or the API key won't be visible.
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text  # noqa: E402

from agentxai.store.trajectory_store import TrajectoryStore  # noqa: E402
from agentxai.xai.accountability import (  # noqa: E402
    _build_explanation_prompt,
    _extract_text,
    _fallback_explanation,
)
from agentxai._llm_factory import build_gemini_llm  # noqa: E402


def _list_task_ids(store: TrajectoryStore, latest_only: bool, all_: bool) -> List[str]:
    with store._engine.connect() as conn:
        rows = conn.execute(
            text("SELECT task_id FROM tasks ORDER BY created_at DESC")
        ).fetchall()
    if not rows:
        return []
    if latest_only:
        return [rows[0][0]]
    if all_:
        return [r[0] for r in rows]
    return []


def regen_one(store: TrajectoryStore, task_id: str, llm) -> Optional[str]:
    record = store.get_full_record(task_id)
    report = record.xai_data.accountability_report
    if report is None:
        print(f"  {task_id[:8]} — no accountability report on file, skipping")
        return None

    old = report.one_line_explanation or ""

    source = "fallback"
    if llm is not None:
        try:
            response = llm.invoke(_build_explanation_prompt(report, record.xai_data))
            llm_text = _extract_text(response).strip()
        except Exception as exc:
            print(f"  {task_id[:8]} — LLM call failed ({exc}); using fallback")
            llm_text = ""
        if llm_text:
            new = llm_text
            source = "llm"
        else:
            print(f"  {task_id[:8]} — LLM returned empty text; using fallback")
            new = _fallback_explanation(report, record.xai_data)
    else:
        new = _fallback_explanation(report, record.xai_data)

    report.one_line_explanation = new
    store.save_accountability_report(report)

    print(f"  {task_id[:8]} ({source})")
    print(f"    OLD: {old[:140]}")
    print(f"    NEW: {new[:140]}")
    return new


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("task_id", nargs="?", help="explicit task id")
    p.add_argument("--latest", action="store_true", help="regen only the most recent task")
    p.add_argument("--all", action="store_true", help="regen every task in the DB")
    p.add_argument("--no-llm", action="store_true",
                   help="skip the LLM and use the templated fallback only")
    args = p.parse_args()

    store = TrajectoryStore()
    if args.task_id:
        task_ids = [args.task_id]
    else:
        task_ids = _list_task_ids(store, args.latest, args.all)

    if not task_ids:
        print("No task id supplied. Use a positional task_id, --latest, or --all.")
        return 1

    llm = None if args.no_llm else build_gemini_llm(model="gemini-2.5-flash-lite", temperature=0)

    print(f"Regenerating explanations for {len(task_ids)} task(s)…")
    for tid in task_ids:
        try:
            regen_one(store, tid, llm)
        except KeyError:
            print(f"  {tid[:8]} — task not found in DB, skipping")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
