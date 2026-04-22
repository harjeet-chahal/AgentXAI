"""
Manual XAI quality review page — Streamlit multi-page app.

Cycles through the 500 manual-review records (deterministic split, seed=42),
showing the patient case, ground truth, system output, accountability report,
causal chain, and one-line explanation for each.  Collects four 1–5 ratings
plus free-text notes and writes them to the manual_reviews SQLite table.

Run alongside the main dashboard:
    streamlit run agentxai/ui/dashboard.py
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------------

_ROOT = pathlib.Path(__file__).resolve().parents[3]   # project root
_DB_PATH = _ROOT / "agentxai" / "data" / "agentxai.db"
sys.path.insert(0, str(_ROOT))

API_BASE = "http://localhost:8000"
HTTP_TIMEOUT = 10

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS manual_reviews (
            review_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id          TEXT NOT NULL,
            plausibility     INTEGER,
            completeness     INTEGER,
            specificity      INTEGER,
            causal_coherence INTEGER,
            notes            TEXT,
            status           TEXT NOT NULL DEFAULT 'reviewed',
            reviewed_at      TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_manual_reviews_task_id "
        "ON manual_reviews(task_id)"
    )
    conn.commit()
    return conn


def _get_done(conn: sqlite3.Connection) -> Dict[str, str]:
    cur = conn.execute("SELECT task_id, status FROM manual_reviews")
    return {row["task_id"]: row["status"] for row in cur.fetchall()}


def _save(
    conn: sqlite3.Connection,
    task_id: str,
    plausibility: Optional[int],
    completeness: Optional[int],
    specificity: Optional[int],
    causal_coherence: Optional[int],
    notes: str,
    status: str,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO manual_reviews
            (task_id, plausibility, completeness, specificity,
             causal_coherence, notes, status, reviewed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_id,
            plausibility, completeness, specificity, causal_coherence,
            notes, status,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading 500 review records…")
def _load_review_records() -> List[Dict[str, Any]]:
    from agentxai.data.load_medqa import load_medqa_us_all, make_splits
    all_records = load_medqa_us_all()
    _, _, review = make_splits(all_records, eval_size=1500, review_size=500, seed=42)
    return review


def _fetch_task(task_id: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE}/tasks/{task_id}", timeout=HTTP_TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _fmt_ts(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
    except Exception:
        return str(ts)


def _render_patient_case(record: Dict[str, Any]) -> None:
    st.markdown("#### Patient Case")
    st.write(record["question"])
    st.markdown("**Options**")
    for letter, text in sorted(record["options"].items()):
        st.markdown(f"**{letter}.** {text}")


def _render_ground_truth(record: Dict[str, Any]) -> None:
    st.markdown("#### Ground Truth")
    correct_letter = record["answer"]
    correct_text = record["options"].get(correct_letter, "—")
    col_a, col_b = st.columns(2)
    col_a.metric("Correct option", correct_letter)
    col_b.metric("Answer text", correct_text[:60] + ("…" if len(correct_text) > 60 else ""))
    st.caption(f"meta_info: `{record.get('meta_info', '—')}`")


def _render_system_output(task_data: Dict[str, Any]) -> None:
    st.markdown("#### System Output")
    sys_out = task_data.get("system_output") or {}
    if not sys_out:
        st.info("No system output recorded for this task.")
        return
    c1, c2, c3 = st.columns(3)
    diag = sys_out.get("final_diagnosis") or "—"
    conf = sys_out.get("confidence")
    correct = sys_out.get("correct")
    c1.metric("Final diagnosis", diag[:50] + ("…" if len(diag) > 50 else ""))
    c2.metric("Confidence", f"{float(conf):.2f}" if conf is not None else "—")
    c3.metric("Correct", "✅" if correct else ("❌" if correct is False else "·"))
    with st.expander("Full system output JSON"):
        st.json(sys_out)


def _render_accountability(task_data: Dict[str, Any]) -> None:
    xai = task_data.get("xai_data") or {}
    report = xai.get("accountability_report") or {}
    if not report:
        st.info("No accountability report recorded.")
        return

    # One-line explanation banner
    one_line = report.get("one_line_explanation") or "—"
    correct = report.get("outcome_correct")
    accent = "#06A77D" if correct else ("#E63946" if correct is False else "#2E86AB")
    st.markdown(
        f"""<div style="padding:14px 18px;background:#f4f6f8;
                border-left:6px solid {accent};border-radius:6px;margin-bottom:10px;">
          <div style="font-size:0.75rem;letter-spacing:0.08em;text-transform:uppercase;
                      color:#888;margin-bottom:4px;">One-line explanation</div>
          <div style="font-size:1.05rem;font-weight:600;color:#1b1b1b;">{one_line}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Causal chain
    chain: List[str] = report.get("causal_chain") or []
    trajectory = xai.get("trajectory") or []
    ev_map = {e.get("event_id"): e for e in trajectory}

    with st.expander(f"Causal chain ({len(chain)} steps)", expanded=True):
        if not chain:
            st.caption("No causal chain recorded.")
        else:
            root_id = report.get("root_cause_event_id") or ""
            for i, eid in enumerate(chain):
                ev = ev_map.get(eid) or {}
                is_root = eid == root_id
                prefix = "🔴 " if is_root else f"{i + 1}. "
                label = (
                    f"{prefix}`{eid[:8]}` — "
                    f"**{ev.get('event_type', '?')}** · "
                    f"{ev.get('agent_id', '?')}"
                    + (f" · {_fmt_ts(ev['timestamp'])}" if ev.get("timestamp") else "")
                )
                st.markdown(label)

    # Responsibility scores
    scores: Dict[str, float] = report.get("agent_responsibility_scores") or {}
    if scores:
        with st.expander("Agent responsibility scores"):
            import pandas as pd
            df = (
                pd.DataFrame({"responsibility": scores})
                .sort_values("responsibility", ascending=True)
            )
            try:
                st.bar_chart(df, horizontal=True)
            except TypeError:
                st.bar_chart(df)

    with st.expander("Full accountability report JSON"):
        st.json(report)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def _first_unreviewed(records: List[Dict], done: Dict[str, str]) -> int:
    for i, rec in enumerate(records):
        if rec["task_id"] not in done:
            return i
    return 0


def _advance(records: List[Dict], conn: sqlite3.Connection) -> None:
    done = _get_done(conn)
    current = st.session_state.get("review_idx", 0)
    for delta in range(1, len(records) + 1):
        candidate = (current + delta) % len(records)
        if records[candidate]["task_id"] not in done:
            st.session_state.review_idx = candidate
            return
    st.session_state.review_idx = current  # all done — stay put


# ---------------------------------------------------------------------------
# Rating form
# ---------------------------------------------------------------------------

def _render_rating_form(
    task_id: str,
    conn: sqlite3.Connection,
    done: Dict[str, str],
    records: List[Dict],
) -> None:
    st.divider()
    st.markdown("### Rate this record")

    # Pre-fill from any existing row
    existing: Optional[Dict] = None
    if task_id in done:
        cur = conn.execute(
            "SELECT * FROM manual_reviews WHERE task_id=?", (task_id,)
        )
        row = cur.fetchone()
        if row:
            existing = dict(row)

    def _default(field: str, fallback: int = 3) -> int:
        if existing and existing.get(field) is not None:
            return int(existing[field])
        return fallback

    c1, c2 = st.columns(2)
    with c1:
        plausibility = st.slider(
            "Plausibility",
            1, 5, value=_default("plausibility"),
            key=f"plaus_{task_id}",
            help="How plausible is the system's reasoning given the patient case?",
        )
        completeness = st.slider(
            "Completeness",
            1, 5, value=_default("completeness"),
            key=f"comp_{task_id}",
            help="Does the accountability report cover all key decision factors?",
        )
    with c2:
        specificity = st.slider(
            "Specificity",
            1, 5, value=_default("specificity"),
            key=f"spec_{task_id}",
            help="Is the explanation specific to this case, not generic?",
        )
        causal_coherence = st.slider(
            "Causal coherence",
            1, 5, value=_default("causal_coherence"),
            key=f"cc_{task_id}",
            help="Does the causal chain logically connect cause to outcome?",
        )

    notes = st.text_area(
        "Notes",
        value=existing["notes"] if existing and existing.get("notes") else "",
        key=f"notes_{task_id}",
        placeholder="Optional free-text observations…",
        height=100,
    )

    btn_save, btn_skip, _ = st.columns([1, 1, 4])
    save_clicked = btn_save.button(
        "Save & next", type="primary", use_container_width=True, key=f"save_{task_id}"
    )
    skip_clicked = btn_skip.button(
        "Skip", use_container_width=True, key=f"skip_{task_id}"
    )

    if save_clicked:
        _save(conn, task_id, plausibility, completeness, specificity, causal_coherence, notes, "reviewed")
        _advance(records, conn)
        st.rerun()

    if skip_clicked:
        _save(conn, task_id, None, None, None, None, notes or "", "skipped")
        _advance(records, conn)
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="XAI Review · AgentXAI", layout="wide")
    st.title("XAI Quality Review")

    records = _load_review_records()
    conn = _get_conn()
    done = _get_done(conn)

    n_total = len(records)
    n_reviewed = sum(1 for s in done.values() if s == "reviewed")
    n_skipped = sum(1 for s in done.values() if s == "skipped")
    n_done = n_reviewed + n_skipped

    # Progress bar
    progress_frac = n_done / n_total if n_total else 0.0
    st.progress(
        progress_frac,
        text=(
            f"**{n_reviewed} reviewed · {n_skipped} skipped · "
            f"{n_total - n_done} remaining** (of {n_total})"
        ),
    )

    if n_done == n_total:
        st.success("All 500 records have been reviewed or skipped.")

    # Initialise index to first unreviewed record
    if "review_idx" not in st.session_state:
        st.session_state.review_idx = _first_unreviewed(records, done)

    # Navigation bar
    nav_prev, nav_info, nav_jump, nav_go, nav_next = st.columns([1, 3, 2, 1, 1])

    with nav_prev:
        if st.button("◀ Prev", use_container_width=True):
            st.session_state.review_idx = max(0, st.session_state.review_idx - 1)
            st.rerun()

    with nav_next:
        if st.button("Next ▶", use_container_width=True):
            st.session_state.review_idx = min(n_total - 1, st.session_state.review_idx + 1)
            st.rerun()

    with nav_jump:
        jump_to = st.number_input(
            "Jump to #",
            min_value=1,
            max_value=n_total,
            value=st.session_state.review_idx + 1,
            step=1,
            label_visibility="collapsed",
        )

    with nav_go:
        if st.button("Go", use_container_width=True):
            st.session_state.review_idx = int(jump_to) - 1
            st.rerun()

    idx = st.session_state.review_idx
    record = records[idx]
    task_id = record["task_id"]
    status_badge = ""
    if task_id in done:
        s = done[task_id]
        status_badge = f" · {'✅ reviewed' if s == 'reviewed' else '⏭ skipped'}"

    with nav_info:
        st.markdown(f"**Record {idx + 1} of {n_total}** · `{task_id}`{status_badge}")

    st.divider()

    # Fetch pipeline data (non-blocking — graceful degradation if API is down)
    task_data = _fetch_task(task_id)

    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        _render_patient_case(record)
        st.divider()
        _render_ground_truth(record)

        if task_data:
            st.divider()
            _render_system_output(task_data)
        else:
            st.divider()
            st.caption(
                "System output not available — pipeline may not have been run for this record yet."
            )

    with right_col:
        if task_data:
            _render_accountability(task_data)
        else:
            st.info(
                "Accountability report not available. "
                "Run this record through the pipeline first, or rate based on "
                "the patient case and ground truth alone."
            )

    _render_rating_form(task_id, conn, done, records)


if __name__ == "__main__":
    main()
