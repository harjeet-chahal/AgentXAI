"""
Aggregate manual XAI quality reviews from the manual_reviews SQLite table.

Reads all rows with status='reviewed', computes mean ± std for each rating
dimension (plausibility, completeness, specificity, causal_coherence), and
prints a Markdown summary table.  Optionally writes JSON and Markdown files.

Usage
-----
    python eval/aggregate_manual_reviews.py
    python eval/aggregate_manual_reviews.py --out-md eval/manual_review_summary.md
    python eval/aggregate_manual_reviews.py --out-md eval/s.md --out-json eval/s.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DB_PATH = _ROOT / "agentxai" / "data" / "agentxai.db"

DIMENSIONS = ["plausibility", "completeness", "specificity", "causal_coherence"]
TOTAL_RECORDS = 500


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(variance)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def _load_counts(db_path: pathlib.Path) -> Dict[str, int]:
    """Return {status: count} across all rows in manual_reviews."""
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM manual_reviews GROUP BY status"
        )
        return {row["status"]: row["cnt"] for row in cur.fetchall()}
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()


def load_reviewed_rows(db_path: pathlib.Path) -> List[Dict]:
    """Return all rows with status='reviewed'."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            "Run the review tool first to generate ratings."
        )
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT * FROM manual_reviews WHERE status='reviewed' ORDER BY reviewed_at"
        )
        return [dict(row) for row in cur.fetchall()]
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            "manual_reviews table not found — run the review tool first."
        ) from exc
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(rows: List[Dict]) -> Dict[str, Dict]:
    """Return {dimension: {n, mean, std, median, min, max}} for reviewed rows."""
    results: Dict[str, Dict] = {}
    for dim in DIMENSIONS:
        values = [float(row[dim]) for row in rows if row.get(dim) is not None]
        mean, std = _mean_std(values)
        if values:
            sorted_v = sorted(values)
            n = len(sorted_v)
            mid = n // 2
            median = sorted_v[mid] if n % 2 else (sorted_v[mid - 1] + sorted_v[mid]) / 2
            lo, hi = sorted_v[0], sorted_v[-1]
        else:
            median = lo = hi = float("nan")
        results[dim] = {
            "n": len(values),
            "mean": round(mean, 3),
            "std": round(std, 3),
            "median": round(median, 3),
            "min": int(lo) if not math.isnan(lo) else None,
            "max": int(hi) if not math.isnan(hi) else None,
        }
    return results


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

_BAR_WIDTH = 20

def _ascii_bar(mean: float, lo: float = 1.0, hi: float = 5.0) -> str:
    if math.isnan(mean):
        return "░" * _BAR_WIDTH
    frac = (mean - lo) / (hi - lo)
    filled = round(frac * _BAR_WIDTH)
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


def build_markdown(
    agg: Dict[str, Dict],
    n_reviewed: int,
    n_skipped: int,
    n_total: int,
) -> str:
    n_pending = n_total - n_reviewed - n_skipped
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Manual XAI Quality Review — Summary",
        "",
        f"| Stat | Value |",
        f"|------|-------|",
        f"| Total records | {n_total} |",
        f"| Reviewed | {n_reviewed} |",
        f"| Skipped | {n_skipped} |",
        f"| Pending | {n_pending} |",
        f"| Coverage | {n_reviewed / n_total * 100:.1f}% |",
        "",
        "## Ratings (1–5 scale)",
        "",
        "| Dimension | N | Mean ± Std | Median | Min | Max | Bar |",
        "|-----------|---|-----------|--------|-----|-----|-----|",
    ]

    for dim, s in agg.items():
        mean = s["mean"]
        std = s["std"]
        n = s["n"]
        if math.isnan(mean):
            mean_std_str = "—"
            bar = "░" * _BAR_WIDTH
        else:
            mean_std_str = f"{mean:.3f} ± {std:.3f}"
            bar = _ascii_bar(mean)
        median_str = f"{s['median']:.2f}" if not math.isnan(s["median"]) else "—"
        min_str = str(s["min"]) if s["min"] is not None else "—"
        max_str = str(s["max"]) if s["max"] is not None else "—"
        lines.append(
            f"| {dim} | {n} | {mean_std_str} | {median_str} "
            f"| {min_str} | {max_str} | `{bar}` |"
        )

    lines.extend(["", f"*Generated: {generated}*"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregate manual XAI quality reviews from agentxai.db."
    )
    p.add_argument(
        "--db",
        type=pathlib.Path,
        default=_DB_PATH,
        help=f"Path to agentxai.db (default: {_DB_PATH})",
    )
    p.add_argument(
        "--out-md",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Write Markdown summary to this file.",
    )
    p.add_argument(
        "--out-json",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Write JSON results to this file.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    counts = _load_counts(args.db)
    n_reviewed = counts.get("reviewed", 0)
    n_skipped = counts.get("skipped", 0)

    if n_reviewed == 0:
        print("No reviewed records found in manual_reviews table.")
        print(f"  DB path: {args.db}")
        return

    rows = load_reviewed_rows(args.db)
    agg = aggregate(rows)
    md = build_markdown(agg, n_reviewed, n_skipped, TOTAL_RECORDS)

    print(md)

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
        print(f"\nMarkdown written → {args.out_md}")

    if args.out_json:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_reviewed": n_reviewed,
            "n_skipped": n_skipped,
            "n_total": TOTAL_RECORDS,
            "dimensions": agg,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON written      → {args.out_json}")


if __name__ == "__main__":
    main()
