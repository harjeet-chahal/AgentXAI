"""
Streamlit dashboard — 7 tabs, one per XAI pillar.

Talks to the FastAPI backend over HTTP. All 7 tabs are implemented:
Trajectory, Plans, Tool Provenance, Memory, Communication, Causality,
Accountability.

Run:
    streamlit run agentxai/ui/dashboard.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from difflib import unified_diff
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components


API_BASE = os.environ.get("AGENTXAI_API_URL", "http://localhost:8000")
HTTP_TIMEOUT = 30

# Four distinct, colorblind-friendly hues — one per agent_id.
AGENT_PALETTE = ["#2E86AB", "#E63946", "#06A77D", "#F4A261"]
TAB_LABELS = [
    "Trajectory",
    "Plans",
    "Tool Provenance",
    "Memory",
    "Communication",
    "Causality",
    "Accountability",
]

# ---------------------------------------------------------------------------
# Visual foundation — single CSS injection used across the whole dashboard.
# ---------------------------------------------------------------------------

def inject_css() -> None:
    """Inject the dashboard's design system on every rerun.

    Streamlit rebuilds the page DOM from scratch on every script run, so the
    CSS must be re-emitted each time — guarding with session_state would
    inject only on the first run and leave subsequent reruns unstyled.
    """
    st.markdown(
        """
        <style>
          :root {
            --xai-fg: #1b1b1b;
            --xai-fg-soft: #3a3f47;
            --xai-muted: #6b7280;
            --xai-line: #e6e9ef;
            --xai-card: #ffffff;
            --xai-card-soft: #f7f9fc;
            --xai-accent: #2E86AB;
            --xai-accent-soft: #eaf2f9;
            --xai-success: #06A77D;
            --xai-success-soft: #e7f6f0;
            --xai-error:   #E63946;
            --xai-error-soft: #fce9eb;
            --xai-warn:    #F4A261;
          }

          /* ---------- Hero (case overview top banner) ---------- */
          .xai-hero {
            padding: 34px 38px 28px 38px;
            background: linear-gradient(135deg, #0e2a47 0%, #143d66 55%, #1d527d 100%);
            border-radius: 14px;
            color: #f5f7fa;
            margin-bottom: 22px;
            box-shadow: 0 10px 30px rgba(14, 42, 71, 0.18);
          }
          .xai-hero-eyebrow {
            font-size: 0.74rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #9bbfe0;
            font-weight: 600;
            margin-bottom: 10px;
          }
          .xai-hero-title {
            font-size: 2.1rem;
            font-weight: 700;
            line-height: 1.18;
            color: #fff;
            margin: 0 0 14px 0;
          }
          .xai-hero-meta {
            display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px;
          }

          /* ---------- Pills / badges ---------- */
          .xai-pill {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 500;
            background: rgba(255,255,255,0.10);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.18);
          }
          .xai-pill b { color: #fff; font-weight: 700; }
          .xai-pill-success {
            background: rgba(6,167,125,0.22);
            border-color: rgba(6,167,125,0.45);
            color: #b9f5dd;
          }
          .xai-pill-success b { color: #b9f5dd; }
          .xai-pill-error {
            background: rgba(230,57,70,0.22);
            border-color: rgba(230,57,70,0.45);
            color: #ffc1c6;
          }
          .xai-pill-error b { color: #ffc1c6; }

          /* ---------- Stat cards ---------- */
          .xai-stat-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 12px;
          }
          @media (max-width: 1100px) {
            .xai-stat-grid { grid-template-columns: repeat(3, 1fr); }
          }
          .xai-stat {
            background: var(--xai-card);
            border: 1px solid var(--xai-line);
            border-radius: 10px;
            padding: 14px 16px;
            transition: border-color 120ms ease, transform 120ms ease;
          }
          .xai-stat:hover {
            border-color: #cdd5e0;
            transform: translateY(-1px);
          }
          .xai-stat-label {
            font-size: 0.7rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--xai-muted);
            font-weight: 600;
          }
          .xai-stat-value {
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--xai-fg);
            line-height: 1.1;
            margin-top: 6px;
          }
          .xai-stat-sub {
            font-size: 0.78rem;
            color: var(--xai-muted);
            margin-top: 4px;
          }

          /* ---------- Section header ---------- */
          .xai-section {
            font-size: 0.74rem;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: var(--xai-muted);
            font-weight: 700;
            margin: 22px 0 12px 0;
            display: flex; align-items: center; gap: 8px;
          }
          .xai-section::before {
            content: ""; display: inline-block;
            width: 16px; height: 2px; background: var(--xai-accent);
          }

          /* ---------- Generic content card ---------- */
          .xai-card {
            background: var(--xai-card);
            border: 1px solid var(--xai-line);
            border-radius: 10px;
            padding: 16px 18px;
            height: 100%;
          }
          .xai-card.is-accent { border-left: 4px solid var(--xai-accent); }
          .xai-card.is-success { border-left: 4px solid var(--xai-success); }
          .xai-card.is-error { border-left: 4px solid var(--xai-error); }
          .xai-card-eyebrow {
            font-size: 0.7rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--xai-muted);
            font-weight: 700;
            margin-bottom: 6px;
          }
          .xai-card-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--xai-fg);
            line-height: 1.3;
          }
          .xai-card-sub {
            font-size: 0.86rem;
            color: var(--xai-fg-soft);
            margin-top: 4px;
            line-height: 1.45;
          }

          /* ---------- One-line conclusion card ---------- */
          .xai-conclusion {
            background: var(--xai-card-soft);
            border-left: 6px solid var(--xai-accent);
            border-radius: 10px;
            padding: 18px 22px;
          }
          .xai-conclusion.is-success { border-left-color: var(--xai-success); }
          .xai-conclusion.is-error { border-left-color: var(--xai-error); }
          .xai-conclusion-eyebrow {
            font-size: 0.7rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--xai-muted);
            font-weight: 700;
            margin-bottom: 8px;
          }
          .xai-conclusion-text {
            font-size: 1.08rem;
            font-weight: 600;
            line-height: 1.45;
            color: var(--xai-fg);
          }

          /* ---------- Per-tab summary banner ---------- */
          .xai-summary {
            background: linear-gradient(180deg, #f1f6fb 0%, #eaf2f9 100%);
            border-left: 4px solid var(--xai-accent);
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 22px;
          }
          .xai-summary-eyebrow {
            font-size: 0.7rem;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: var(--xai-accent);
            font-weight: 700;
            margin-bottom: 6px;
          }
          .xai-summary-headline {
            font-size: 0.98rem;
            line-height: 1.5;
            color: var(--xai-fg);
          }
          .xai-summary ul {
            margin: 10px 0 0 0;
            padding-left: 20px;
            font-size: 0.9rem;
            line-height: 1.55;
            color: var(--xai-fg-soft);
          }
          .xai-summary code {
            background: rgba(46,134,171,0.10);
            padding: 1px 6px;
            border-radius: 4px;
            font-size: 0.85em;
            color: #1d527d;
          }
          .xai-summary b { color: var(--xai-fg); }

          /* ---------- Slim case strip (top of pillar pages) ---------- */
          .xai-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 12px 18px;
            background: #fafbfc;
            border: 1px solid var(--xai-line);
            border-radius: 10px;
            margin-bottom: 18px;
          }
          .xai-strip-left { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
          .xai-strip-id {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.82rem;
            color: var(--xai-muted);
          }
          .xai-strip-dx {
            font-weight: 700;
            color: var(--xai-fg);
            font-size: 0.98rem;
          }
          .xai-strip-meta { color: var(--xai-muted); font-size: 0.85rem; }
          .xai-strip-badge {
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
          }
          .xai-strip-badge.is-success {
            background: var(--xai-success-soft); color: var(--xai-success);
          }
          .xai-strip-badge.is-error {
            background: var(--xai-error-soft); color: var(--xai-error);
          }
          .xai-strip-badge.is-neutral {
            background: var(--xai-accent-soft); color: var(--xai-accent);
          }

          /* ---------- Responsibility distribution ---------- */
          .xai-resp-row { display: flex; align-items: center; gap: 12px; margin: 6px 0; }
          .xai-resp-name {
            width: 150px;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.85rem;
            color: var(--xai-fg);
          }
          .xai-resp-bar {
            flex: 1;
            height: 10px;
            background: #eef0f4;
            border-radius: 999px;
            overflow: hidden;
          }
          .xai-resp-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #2E86AB 0%, #1d527d 100%);
            border-radius: 999px;
          }
          .xai-resp-val {
            width: 56px;
            text-align: right;
            font-variant-numeric: tabular-nums;
            font-weight: 700;
            color: var(--xai-fg);
            font-size: 0.92rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Per-tab summarizers — return a (headline_html, bullets) tuple. The headline
# is one short sentence with inline <b> / <code>; bullets are short richer
# breakdowns rendered as a styled list inside the same banner.
# ---------------------------------------------------------------------------

Summary = Tuple[str, List[str]]


def _sort_by_count(d: Dict[str, int], reverse: bool = True) -> List[Tuple[str, int]]:
    return sorted(d.items(), key=lambda kv: kv[1], reverse=reverse)


def _summarize_trajectory(events: List[Dict[str, Any]]) -> Summary:
    if not events:
        return ("No trajectory events recorded for this task.", [])

    by_agent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ev in events:
        by_agent[ev.get("agent_id") or "unknown"].append(ev)

    times = [
        ev.get("timestamp") for ev in events
        if isinstance(ev.get("timestamp"), (int, float))
    ]
    span_str = ""
    if len(times) >= 2:
        span_str = f", spanning <b>{(max(times) - min(times)):.1f}s</b>"

    sorted_events = sorted(events, key=lambda e: e.get("timestamp", 0))
    final = sorted_events[-1]
    final_label = final.get("action") or final.get("event_type") or "event"

    headline = (
        f"<b>{len(events)} events</b> across <b>{len(by_agent)} agent(s)</b>{span_str}, "
        f"closing on <code>{final_label}</code>."
    )

    bullets = []
    for agent_id, ev_list in sorted(
        by_agent.items(), key=lambda kv: len(kv[1]), reverse=True
    ):
        type_counts: Dict[str, int] = defaultdict(int)
        for ev in ev_list:
            type_counts[ev.get("event_type") or "event"] += 1
        breakdown = ", ".join(
            f"{c}× <code>{t}</code>" for t, c in _sort_by_count(type_counts)
        )
        bullets.append(
            f"<code>{agent_id}</code> — <b>{len(ev_list)} events</b> "
            f"({breakdown})"
        )
    return headline, bullets


def _summarize_plans(plans: List[Dict[str, Any]]) -> Summary:
    if not plans:
        return ("No plans recorded for this task.", [])

    by_agent: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"plans": 0, "intended": 0, "actual": 0, "deviations": 0}
    )
    for p in plans:
        a = p.get("agent_id") or "unknown"
        by_agent[a]["plans"] += 1
        by_agent[a]["intended"] += len(p.get("intended_actions") or [])
        by_agent[a]["actual"] += len(p.get("actual_actions") or [])
        by_agent[a]["deviations"] += len(p.get("deviations") or [])

    total_dev = sum(v["deviations"] for v in by_agent.values())
    if total_dev == 0:
        headline = (
            f"<b>{len(plans)} plan(s)</b> across "
            f"<b>{len(by_agent)} agent(s)</b> — every agent followed its plan exactly."
        )
    else:
        deviating = ", ".join(
            f"<code>{a}</code> ({v['deviations']})"
            for a, v in by_agent.items() if v["deviations"]
        )
        headline = (
            f"<b>{len(plans)} plan(s)</b> across <b>{len(by_agent)} agent(s)</b>, "
            f"with <b>{total_dev} deviation(s)</b> from {deviating}."
        )

    bullets = []
    for agent_id, v in sorted(
        by_agent.items(), key=lambda kv: kv[1]["deviations"], reverse=True
    ):
        line = (
            f"<code>{agent_id}</code> — planned <b>{v['intended']}</b>, "
            f"executed <b>{v['actual']}</b>"
        )
        if v["deviations"]:
            line += (
                f", <b style='color:#E63946;'>{v['deviations']} deviation(s)</b>"
            )
        else:
            line += " (no deviations)"
        bullets.append(line)
    return headline, bullets


def _summarize_tools(tool_calls: List[Dict[str, Any]]) -> Summary:
    if not tool_calls:
        return ("No tool calls recorded for this task.", [])

    by_tool: Dict[str, int] = defaultdict(int)
    impact_by_tool: Dict[str, float] = defaultdict(float)
    for c in tool_calls:
        name = c.get("tool_name") or "?"
        by_tool[name] += 1
        impact_by_tool[name] = max(
            impact_by_tool[name], float(c.get("downstream_impact_score") or 0)
        )
    total_ms = sum(float(c.get("duration_ms") or 0) for c in tool_calls)

    ranked = sorted(
        tool_calls,
        key=lambda c: float(c.get("downstream_impact_score") or 0),
        reverse=True,
    )
    top = ranked[0]
    top_name = top.get("tool_name") or "?"
    top_score = float(top.get("downstream_impact_score") or 0)
    top_caller = top.get("called_by") or "?"

    headline = (
        f"<b>{len(tool_calls)} tool call(s)</b> across "
        f"<b>{len(by_tool)} distinct tool(s)</b> "
        f"(<b>{total_ms:.0f}ms</b> wall time). "
        f"Highest impact: <code>{top_name}</code> via <code>{top_caller}</code> "
        f"at <b>{top_score:.2f}</b>."
    )

    bullets = []
    for c in ranked[: min(3, len(ranked))]:
        name = c.get("tool_name") or "?"
        caller = c.get("called_by") or "?"
        score = float(c.get("downstream_impact_score") or 0)
        dur = float(c.get("duration_ms") or 0)
        bullets.append(
            f"<code>{name}</code> via <code>{caller}</code> — "
            f"impact <b>{score:.2f}</b>, {dur:.0f}ms"
        )
    if len(ranked) > 3:
        bullets.append(
            f"<i>+ {len(ranked) - 3} more call(s) below this in impact ranking.</i>"
        )
    return headline, bullets


def _summarize_memory(memory_diffs: List[Dict[str, Any]]) -> Summary:
    if not memory_diffs:
        return ("No memory activity recorded for this task.", [])
    writes = [
        m for m in memory_diffs
        if (m.get("operation") or "").lower() == "write"
    ]
    if not writes:
        return (
            f"<b>{len(memory_diffs)} memory event(s)</b> recorded, "
            "but no writes — memory was read-only on this case.",
            [],
        )

    by_key: Dict[str, int] = defaultdict(int)
    by_agent: Dict[str, int] = defaultdict(int)
    key_to_agents: Dict[str, set] = defaultdict(set)
    for m in writes:
        k = m.get("key") or "?"
        a = m.get("agent_id") or "unknown"
        by_key[k] += 1
        by_agent[a] += 1
        key_to_agents[k].add(a)

    top_key, top_count = max(by_key.items(), key=lambda kv: kv[1])
    headline = (
        f"<b>{len(writes)} write(s)</b> across <b>{len(by_agent)} agent(s)</b> "
        f"into <b>{len(by_key)} key(s)</b>. "
        f"Most-touched: <code>{top_key}</code> ({top_count} writes)."
    )

    bullets = []
    for k, n in _sort_by_count(by_key)[:4]:
        who = ", ".join(f"<code>{a}</code>" for a in sorted(key_to_agents[k]))
        bullets.append(f"<code>{k}</code> — <b>{n} write(s)</b> by {who}")
    if len(by_key) > 4:
        bullets.append(f"<i>+ {len(by_key) - 4} more key(s).</i>")
    return headline, bullets


def _summarize_messages(messages: List[Dict[str, Any]]) -> Summary:
    if not messages:
        return ("No inter-agent messages recorded for this task.", [])

    n = len(messages)
    acted = sum(1 for m in messages if m.get("acted_upon"))
    parties = (
        {m.get("sender") or "unknown" for m in messages}
        | {m.get("receiver") or "unknown" for m in messages}
    )

    pair_counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "acted": 0}
    )
    for m in messages:
        key = (m.get("sender") or "unknown", m.get("receiver") or "unknown")
        pair_counts[key]["total"] += 1
        if m.get("acted_upon"):
            pair_counts[key]["acted"] += 1

    pct = (acted / n * 100) if n else 0
    headline = (
        f"<b>{n} message(s)</b> between <b>{len(parties)} agent(s)</b> — "
        f"<b style='color:#06A77D;'>{acted} acted upon</b>, "
        f"<b style='color:#6b7280;'>{n - acted} ignored</b> "
        f"(<b>{pct:.0f}%</b> effective)."
    )

    bullets = []
    sorted_pairs = sorted(
        pair_counts.items(), key=lambda kv: kv[1]["total"], reverse=True
    )
    for (s, r), counts in sorted_pairs[:4]:
        bullets.append(
            f"<code>{s}</code> → <code>{r}</code> — "
            f"<b>{counts['total']} msg(s)</b>, "
            f"<b style='color:#06A77D;'>{counts['acted']}</b> acted upon"
        )
    if len(sorted_pairs) > 4:
        bullets.append(f"<i>+ {len(sorted_pairs) - 4} more channel(s).</i>")
    return headline, bullets


def _summarize_causality(
    causal_graph: Dict[str, Any], report: Optional[Dict[str, Any]]
) -> Summary:
    nodes = (causal_graph or {}).get("nodes") or []
    edges = (causal_graph or {}).get("edges") or []
    if not nodes and not edges:
        return ("No causal graph recorded for this task.", [])

    node_set = set(nodes)
    for e in edges:
        for nid in (e.get("cause_event_id"), e.get("effect_event_id")):
            if nid:
                node_set.add(nid)

    type_counts: Dict[str, int] = defaultdict(int)
    type_strength: Dict[str, float] = defaultdict(float)
    for e in edges:
        t = e.get("causal_type") or "?"
        type_counts[t] += 1
        type_strength[t] += float(e.get("causal_strength") or 0)

    head_parts = [
        f"<b>{len(node_set)} node(s)</b>, <b>{len(edges)} edge(s)</b>."
    ]
    if edges:
        strongest = max(
            edges, key=lambda e: float(e.get("causal_strength") or 0)
        )
        head_parts.append(
            f"Strongest link: <code>{strongest.get('causal_type', '?')}</code> "
            f"at <b>{float(strongest.get('causal_strength') or 0):.2f}</b>."
        )
    root = ((report or {}).get("root_cause_event_id") or "").strip()
    if root:
        head_parts.append(
            f"Root cause: <code>{root[:8]}</code> (highlighted in red)."
        )
    headline = " ".join(head_parts)

    bullets = []
    for t, c in _sort_by_count(type_counts):
        avg = (type_strength[t] / c) if c else 0
        bullets.append(
            f"<code>{t}</code> — <b>{c} edge(s)</b>, mean strength <b>{avg:.2f}</b>"
        )
    return headline, bullets


def _summarize_accountability(report: Optional[Dict[str, Any]]) -> Summary:
    if not report:
        return ("No accountability report recorded for this task.", [])
    correct = report.get("outcome_correct")
    badge = (
        "<b style='color:#06A77D;'>✅ correct</b>" if correct
        else (
            "<b style='color:#E63946;'>❌ incorrect</b>"
            if correct is False else "—"
        )
    )
    explanation = report.get("one_line_explanation") or ""
    explanation_html = (
        f" <i>{explanation}</i>" if explanation else ""
    )
    headline = f"Outcome: {badge}.{explanation_html}"

    bullets = []
    scores = report.get("agent_responsibility_scores") or {}
    if scores:
        for agent_id, score in sorted(
            scores.items(), key=lambda kv: float(kv[1]), reverse=True
        ):
            bullets.append(
                f"<code>{agent_id}</code> — responsibility <b>{float(score):.2f}</b>"
            )
    chain = report.get("causal_chain") or []
    if chain:
        bullets.append(
            f"Causal chain length: <b>{len(chain)}</b> events linking inputs to outcome."
        )
    return headline, bullets


def render_tab_summary(
    tab_name: str, summary: Summary
) -> None:
    """Per-task summary banner shown at the top of each pillar tab."""
    headline, bullets = summary
    if not headline and not bullets:
        return
    bullets_html = ""
    if bullets:
        items = "".join(f"<li>{b}</li>" for b in bullets)
        bullets_html = f"<ul>{items}</ul>"
    st.markdown(
        f"""
        <div class="xai-summary">
          <div class="xai-summary-eyebrow">{tab_name} · this case</div>
          <div class="xai-summary-headline">{headline}</div>
          {bullets_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Per-case overview (the "large summary" landing page for each task)
# ---------------------------------------------------------------------------

def _truncate_case(text: str, limit: int = 700) -> Tuple[str, bool]:
    text = (text or "").strip()
    if len(text) <= limit:
        return text, False
    cut = text.rfind(". ", 0, limit)
    if cut < int(limit * 0.6):
        cut = limit
    return text[: cut + 1].rstrip() + " …", True


def _stat_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="xai-stat-sub">{sub}</div>' if sub else ""
    return (
        '<div class="xai-stat">'
        f'<div class="xai-stat-label">{label}</div>'
        f'<div class="xai-stat-value">{value}</div>'
        f'{sub_html}'
        '</div>'
    )


def _section(title: str) -> None:
    st.markdown(
        f'<div class="xai-section">{title}</div>', unsafe_allow_html=True
    )


def _info_card(eyebrow: str, title: str, sub: str = "", variant: str = "accent") -> str:
    sub_html = f'<div class="xai-card-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="xai-card is-{variant}">'
        f'<div class="xai-card-eyebrow">{eyebrow}</div>'
        f'<div class="xai-card-title">{title}</div>'
        f'{sub_html}'
        '</div>'
    )


def render_case_overview(record: Dict[str, Any]) -> None:
    """Large per-case summary — the landing page for the selected task."""
    task_id = record.get("task_id", "—")
    system_output = record.get("system_output") or {}
    ground_truth = record.get("ground_truth") or {}
    input_data = record.get("input") or {}
    xai = record.get("xai_data") or {}
    report = xai.get("accountability_report") or {}

    final_diagnosis = system_output.get("final_diagnosis") or "—"
    correct_answer = ground_truth.get("correct_answer") or "—"
    correct = system_output.get("correct")
    if correct is None:
        correct = report.get("outcome_correct")
    confidence = system_output.get("confidence")
    explanation = report.get("one_line_explanation") or "—"
    conf_str = (
        f"{float(confidence):.2f}"
        if isinstance(confidence, (int, float)) else "—"
    )
    correct_pill = (
        '<span class="xai-pill xai-pill-success">✓ <b>Correct</b></span>'
        if correct
        else (
            '<span class="xai-pill xai-pill-error">✗ <b>Incorrect</b></span>'
            if correct is False else
            '<span class="xai-pill"><b>—</b></span>'
        )
    )

    # --- Hero --------------------------------------------------------------
    st.markdown(
        f"""
        <div class="xai-hero">
          <div class="xai-hero-eyebrow">Case overview · task {task_id[:12]}</div>
          <h1 class="xai-hero-title">{final_diagnosis}</h1>
          <div class="xai-hero-meta">
            {correct_pill}
            <span class="xai-pill">Ground truth: <b>&nbsp;{correct_answer}</b></span>
            <span class="xai-pill">Confidence: <b>&nbsp;{conf_str}</b></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Aggregate counts (right under hero) -------------------------------
    trajectory = xai.get("trajectory", []) or []
    plans = xai.get("plans", []) or []
    tool_calls = xai.get("tool_calls", []) or []
    memory_diffs = xai.get("memory_diffs", []) or []
    messages = xai.get("messages", []) or []
    causal_graph = xai.get("causal_graph", {}) or {}
    edges = causal_graph.get("edges") or []
    writes = [
        m for m in memory_diffs
        if (m.get("operation") or "").lower() == "write"
    ]
    deviations_total = sum(len(p.get("deviations") or []) for p in plans)
    acted_msgs = sum(1 for m in messages if m.get("acted_upon"))
    agents_seen = (
        {ev.get("agent_id") for ev in trajectory if ev.get("agent_id")}
        | {p.get("agent_id") for p in plans if p.get("agent_id")}
    )

    cards = [
        _stat_card(
            "Events", str(len(trajectory)),
            f"{len(agents_seen)} agents involved" if agents_seen else "",
        ),
        _stat_card(
            "Plans", str(len(plans)),
            f"{deviations_total} deviation(s)" if deviations_total else "no deviations",
        ),
        _stat_card("Tool calls", str(len(tool_calls)), ""),
        _stat_card("Memory writes", str(len(writes)), ""),
        _stat_card(
            "Messages", str(len(messages)),
            f"{acted_msgs} acted upon" if messages else "",
        ),
        _stat_card("Causal edges", str(len(edges)), ""),
    ]
    st.markdown(
        '<div class="xai-stat-grid">' + "".join(cards) + '</div>',
        unsafe_allow_html=True,
    )

    # --- Patient case + answer reasoning -----------------------------------
    _section("The case")
    patient_case = input_data.get("patient_case") or ""
    options = input_data.get("answer_options") or {}
    options = options if isinstance(options, dict) else {}

    case_left, case_right = st.columns([3, 2])

    with case_left:
        st.markdown(
            '<div class="xai-card-eyebrow">Patient case</div>',
            unsafe_allow_html=True,
        )
        if patient_case.strip():
            shown, was_truncated = _truncate_case(patient_case, limit=700)
            st.write(shown)
            if was_truncated:
                with st.expander("Show full case"):
                    st.write(patient_case)
        else:
            st.caption("No patient-case text recorded for this task.")

        if options:
            st.markdown(
                '<div class="xai-card-eyebrow" style="margin-top:14px;">'
                'Answer options</div>',
                unsafe_allow_html=True,
            )
            for k, v in options.items():
                if not v:
                    continue
                is_truth = (k == correct_answer)
                marker = "🟢 " if is_truth else ""
                st.markdown(
                    f"- **{k}.** {marker}{v}"
                    + ("  _← ground truth_" if is_truth else "")
                )

    with case_right:
        variant = (
            "success" if correct
            else ("error" if correct is False else "accent")
        )
        st.markdown(
            f"""
            <div class="xai-conclusion is-{variant}">
              <div class="xai-conclusion-eyebrow">System's one-line explanation</div>
              <div class="xai-conclusion-text">{explanation}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        gt_explanation = ground_truth.get("explanation") or ""
        if gt_explanation:
            with st.expander("Ground-truth explanation"):
                st.write(gt_explanation)

    # --- Responsibility distribution ---------------------------------------
    scores: Dict[str, float] = report.get("agent_responsibility_scores") or {}
    if scores:
        _section("Agent responsibility")
        max_score = max(float(s) for s in scores.values()) or 1.0
        rows = []
        for agent_id, score in sorted(
            scores.items(), key=lambda kv: float(kv[1]), reverse=True
        ):
            pct = (float(score) / max_score) * 100
            rows.append(
                '<div class="xai-resp-row">'
                f'<div class="xai-resp-name">{agent_id}</div>'
                '<div class="xai-resp-bar">'
                f'<div class="xai-resp-bar-fill" style="width:{pct:.1f}%;"></div>'
                '</div>'
                f'<div class="xai-resp-val">{float(score):.2f}</div>'
                '</div>'
            )
        st.markdown(
            '<div class="xai-card">' + "".join(rows) + '</div>',
            unsafe_allow_html=True,
        )

    # --- Key attributions (4 cards) ----------------------------------------
    _section("Key attributions")
    root = report.get("root_cause_event_id") or ""
    impactful_tool_id = report.get("most_impactful_tool_call_id") or ""
    influential_msg_id = report.get("most_influential_message_id") or ""
    ev_map = {e.get("event_id"): e for e in trajectory}

    # Top agent card. When responsibilities are tied or near-tied (within
    # 0.05), credit the whole tied group rather than arbitrarily naming
    # one — otherwise the headline misrepresents the actual attribution.
    if scores:
        sorted_scores = sorted(
            scores.items(), key=lambda kv: float(kv[1]), reverse=True
        )
        top_score = float(sorted_scores[0][1])
        tied_agents = [
            a for a, s in sorted_scores
            if top_score - float(s) < 0.05
        ]
        if len(tied_agents) > 1 and len(tied_agents) == len(sorted_scores):
            # Full tie across every recorded specialist.
            label = " = ".join(tied_agents) + f" · {top_score:.2f} each"
            sub = (
                f"All {len(tied_agents)} specialists contributed equally — "
                "perturbing any one of them produced an identical "
                "counterfactual outcome change."
            )
        elif len(tied_agents) > 1:
            label = " = ".join(tied_agents) + f" · {top_score:.2f}"
            sub = (
                f"{len(tied_agents)} agents tied for highest responsibility "
                "(within 0.05). Lower-ranked: "
                + ", ".join(
                    f"{a} ({float(s):.2f})"
                    for a, s in sorted_scores[len(tied_agents):]
                )
            )
        else:
            top_agent = sorted_scores[0][0]
            label = f"{top_agent} · {top_score:.2f}"
            sub = "Highest counterfactual responsibility for the final outcome."
        agent_card = _info_card(
            "Top responsible agent", label, sub, variant="accent",
        )
    else:
        agent_card = _info_card(
            "Top responsible agent", "Not recorded", "", variant="accent"
        )

    # Root cause card
    if root:
        ev = ev_map.get(root)
        if ev:
            sub = (
                f"{ev.get('event_type', '?')} from {ev.get('agent_id', '?')}"
                + (f" — {ev['action']}" if ev.get("action") else "")
            )
            root_card = _info_card(
                "Root-cause event", root[:8], sub, variant="error"
            )
        else:
            root_card = _info_card(
                "Root-cause event", root[:8], "", variant="error"
            )
    else:
        root_card = _info_card(
            "Root-cause event", "Not recorded", "", variant="accent"
        )

    # Top tool card. Distinguish three cases:
    #   1. No tool calls at all → "Not recorded".
    #   2. Tool surfaced AND its impact is > 0 → green-bordered, real metric.
    #   3. Tool surfaced BUT every tool's measured impact was 0 → still show
    #      the representative tool, but flag the no-measurable-impact reality
    #      so the user understands what 0.00 means.
    if impactful_tool_id:
        tool = next(
            (t for t in tool_calls if t.get("tool_call_id") == impactful_tool_id),
            None,
        )
        if tool:
            impact = float(tool.get("downstream_impact_score") or 0)
            if impact > 0:
                tool_card = _info_card(
                    "Most-impactful tool call",
                    tool.get("tool_name", "?"),
                    (
                        f"Called by {tool.get('called_by', '?')} · "
                        f"impact {impact:.2f}"
                    ),
                    variant="success",
                )
            else:
                tool_card = _info_card(
                    "Most-impactful tool call",
                    f"{tool.get('tool_name', '?')} · impact 0.00",
                    (
                        f"No measurable counterfactual impact — patching any "
                        f"single tool's output didn't change the diagnosis. "
                        f"The LLM-driven specialists were robust to single-tool "
                        f"ablation; impact is concentrated at the agent-memory "
                        f"level (see agent responsibility above)."
                    ),
                    variant="accent",
                )
        else:
            tool_card = _info_card(
                "Most-impactful tool call", impactful_tool_id[:8], "", variant="accent"
            )
    else:
        tool_card = _info_card(
            "Most-impactful tool call", "No tool calls recorded", "", variant="accent"
        )

    # Top message card. Three honest cases for the sub-text:
    #   1. behavior_change_description recorded → use it verbatim.
    #   2. acted_upon flag set → "Acted upon by receiver."
    #   3. neither → "Highest-weighted message; no measurable behavior change."
    # The previous fallback always said "Acted upon by receiver" which
    # contradicted the messages-stat strip when acted_upon was actually False.
    if influential_msg_id:
        msg = next(
            (m for m in messages if m.get("message_id") == influential_msg_id),
            None,
        )
        if msg:
            description = msg.get("behavior_change_description") or ""
            acted = bool(msg.get("acted_upon"))
            if description:
                sub = description
            elif acted:
                sub = "Acted upon by receiver."
            else:
                sub = (
                    "Highest-weighted message by counterfactual delta — "
                    "no measurable behavior change recorded."
                )
            msg_card = _info_card(
                "Most-influential message",
                f"{msg.get('sender', '?')} → {msg.get('receiver', '?')}",
                sub,
                variant="accent",
            )
        else:
            msg_card = _info_card(
                "Most-influential message", influential_msg_id[:8], "", variant="accent"
            )
    else:
        msg_card = _info_card(
            "Most-influential message", "Not recorded", "", variant="accent"
        )

    c1, c2 = st.columns(2)
    c1.markdown(agent_card, unsafe_allow_html=True)
    c2.markdown(root_card, unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    c3.markdown(tool_card, unsafe_allow_html=True)
    c4.markdown(msg_card, unsafe_allow_html=True)

    if report.get("plan_deviation_summary"):
        _section("Plan deviation summary")
        st.write(report["plan_deviation_summary"])

    # --- CTA ----------------------------------------------------------------
    st.markdown("<div style='margin-top:26px;'></div>", unsafe_allow_html=True)
    cta_l, cta_c, cta_r = st.columns([1, 1.4, 1])
    with cta_c:
        if st.button(
            "Continue to the 7 pillar views  →",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["view"] = "explore"
            st.rerun()


def render_case_strip(record: Dict[str, Any]) -> None:
    """Slim case context strip shown at the top of every pillar tab."""
    task_id = record.get("task_id", "—")
    system_output = record.get("system_output") or {}
    ground_truth = record.get("ground_truth") or {}
    final_diagnosis = system_output.get("final_diagnosis") or "—"
    correct_answer = ground_truth.get("correct_answer") or "—"
    correct = system_output.get("correct")
    confidence = system_output.get("confidence")
    conf_str = (
        f"{float(confidence):.2f}"
        if isinstance(confidence, (int, float)) else "—"
    )
    if correct:
        badge = '<span class="xai-strip-badge is-success">✓ Correct</span>'
    elif correct is False:
        badge = '<span class="xai-strip-badge is-error">✗ Incorrect</span>'
    else:
        badge = '<span class="xai-strip-badge is-neutral">—</span>'

    # Truncate long diagnoses to keep the strip on one line
    dx_short = (
        final_diagnosis if len(final_diagnosis) <= 70
        else final_diagnosis[:67] + "…"
    )

    st.markdown(
        f"""
        <div class="xai-strip">
          <div class="xai-strip-left">
            <span class="xai-strip-id">task {task_id[:8]}</span>
            <span class="xai-strip-dx">{dx_short}</span>
            <span class="xai-strip-meta">vs. truth <b>{correct_answer}</b></span>
            <span class="xai-strip-meta">confidence <b>{conf_str}</b></span>
          </div>
          <div>{badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _url(path: str) -> str:
    return urljoin(API_BASE.rstrip("/") + "/", path.lstrip("/"))


def api_get(path: str, **params: Any) -> Any:
    r = requests.get(_url(path), params=params or None, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: Dict[str, Any]) -> Any:
    r = requests.post(_url(path), json=payload, timeout=HTTP_TIMEOUT * 10)
    r.raise_for_status()
    return r.json()


# Cached read wrappers. Streamlit re-runs the whole script on every UI
# interaction; without these, every click re-fetches the full XAI payload.
#
#   - cached_task_record: per-task records are immutable once stored, so we
#     cache them indefinitely keyed by task_id. ~zero cost on subsequent
#     interactions (expanders, tab switches) within the same task.
#   - cached_task_list: changes when a new task is run. Short 30s TTL plus
#     an explicit clear() on Refresh and after a successful run.

@st.cache_data(ttl=None, show_spinner=False)
def cached_task_record(task_id: str) -> Dict[str, Any]:
    return api_get(f"/tasks/{task_id}")


@st.cache_data(ttl=30, show_spinner=False)
def cached_task_list(per_page: int = 200) -> Dict[str, Any]:
    return api_get("/tasks", per_page=per_page)


def invalidate_task_caches() -> None:
    """Bust the cached reads — call after Refresh / new-task submit."""
    cached_task_list.clear()
    cached_task_record.clear()


# ---------------------------------------------------------------------------
# Sidebar — task selector + "Run new task"
# ---------------------------------------------------------------------------

def render_sidebar() -> Optional[str]:
    st.sidebar.title("AgentXAI")
    st.sidebar.caption(f"API: `{API_BASE}`")

    # View toggle — per-case overview vs. detailed pillar tabs.
    current_view = st.session_state.get("view", "landing")
    nav_a, nav_b = st.sidebar.columns(2)
    if nav_a.button(
        "📋 Summary",
        use_container_width=True,
        type=("primary" if current_view == "landing" else "secondary"),
        help="Per-case overview: what happened on this task.",
    ):
        st.session_state["view"] = "landing"
        st.rerun()
    if nav_b.button(
        "🔬 Pillars",
        use_container_width=True,
        type=("primary" if current_view == "explore" else "secondary"),
        help="The 7 detailed pillar views for this task.",
    ):
        st.session_state["view"] = "explore"
        st.rerun()

    st.sidebar.divider()

    # Task list (cached, short TTL — see cached_task_list)
    try:
        data = cached_task_list(200)
        items: List[Dict[str, Any]] = data.get("items", [])
    except Exception as e:
        st.sidebar.error(f"Could not fetch /tasks: {e}")
        items = []

    def _label(item: Dict[str, Any]) -> str:
        ok = item.get("outcome_correct")
        badge = "✅" if ok else ("❌" if ok is False else "·")
        outcome = item.get("final_outcome") or "—"
        return f"{badge} {item['task_id'][:8]} · {outcome[:40]}"

    labels = [_label(it) for it in items]
    by_label = {lbl: it["task_id"] for lbl, it in zip(labels, items)}

    selected_label = st.sidebar.selectbox(
        "Task",
        options=labels,
        index=0 if labels else None,
        placeholder="No tasks yet" if not labels else "Pick a task",
    )
    selected_task_id = by_label.get(selected_label) if selected_label else None

    if st.sidebar.button("↻ Refresh", use_container_width=True):
        invalidate_task_caches()
        st.rerun()

    st.sidebar.divider()

    # Run new task
    with st.sidebar.expander("▶ Run new task", expanded=not items):
        st.caption(
            "Paste a MedQA record as JSON. Flat MedQA shape: "
            "`question` (the patient case), `options` (A–E), `answer_idx` "
            "(letter)."
        )
        default_record = json.dumps(
            {
                "question": "",
                "options": {"A": "", "B": "", "C": "", "D": "", "E": ""},
                "answer_idx": "A",
                "meta_info": "step1",
            },
            indent=2,
        )
        record_text = st.text_area(
            "MedQA record (JSON)",
            value=default_record,
            height=260,
            label_visibility="collapsed",
        )
        if st.button("Run task", type="primary", use_container_width=True):
            try:
                record = json.loads(record_text)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                return selected_task_id

            with st.spinner("Running pipeline…"):
                try:
                    resp = api_post("/tasks/run", {"record": record})
                except Exception as e:
                    st.error(f"POST /tasks/run failed: {e}")
                    return selected_task_id

            new_id = resp.get("task_id", "?")
            st.success(f"Task {new_id[:8]} queued")
            st.session_state["preferred_task_id"] = new_id
            invalidate_task_caches()
            st.rerun()

    # Honor a newly-run task on next render.
    preferred = st.session_state.pop("preferred_task_id", None)
    if preferred and preferred in {it["task_id"] for it in items}:
        return preferred
    return selected_task_id


# ---------------------------------------------------------------------------
# Tab 1 — Trajectory
# ---------------------------------------------------------------------------

def _agent_color_map(events: List[Dict[str, Any]]) -> Dict[str, str]:
    agents: List[str] = []
    for ev in events:
        aid = ev.get("agent_id") or "unknown"
        if aid not in agents:
            agents.append(aid)
    return {aid: AGENT_PALETTE[i % len(AGENT_PALETTE)] for i, aid in enumerate(agents)}


def _fmt_ts(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S.%f")[:-3]
    except (TypeError, ValueError, OSError):
        return str(ts)


def _state_diff(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    b = json.dumps(before or {}, indent=2, sort_keys=True, default=str).splitlines()
    a = json.dumps(after or {}, indent=2, sort_keys=True, default=str).splitlines()
    diff = unified_diff(b, a, fromfile="state_before", tofile="state_after", lineterm="")
    return "\n".join(diff)


def render_trajectory_tab(trajectory: List[Dict[str, Any]]) -> None:
    render_tab_summary("Trajectory", _summarize_trajectory(trajectory))
    if not trajectory:
        st.info("No trajectory events recorded for this task.")
        return

    events = sorted(trajectory, key=lambda e: e.get("timestamp", 0))
    colors = _agent_color_map(events)

    # Legend
    st.caption("Agents")
    legend_cols = st.columns(min(len(colors), 4) or 1)
    for i, (aid, color) in enumerate(colors.items()):
        with legend_cols[i % len(legend_cols)]:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;'>"
                f"<span style='display:inline-block;width:14px;height:14px;"
                f"border-radius:50%;background:{color};'></span>"
                f"<span style='font-size:0.9rem;'>{aid}</span></div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # Vertical timeline — one row per event. The colored left bar is the
    # per-agent color; clicking the expander header reveals the JSON diff.
    for idx, ev in enumerate(events):
        agent_id = ev.get("agent_id") or "unknown"
        color = colors.get(agent_id, "#888")
        ts = _fmt_ts(ev.get("timestamp", 0))
        event_type = ev.get("event_type") or "event"
        action = ev.get("action") or ""
        outcome = ev.get("outcome") or ""

        bar_col, body_col = st.columns([0.04, 0.96])
        with bar_col:
            st.markdown(
                f"<div style='width:6px;background:{color};border-radius:3px;"
                f"height:64px;margin-top:6px;'></div>",
                unsafe_allow_html=True,
            )
        with body_col:
            header = (
                f"**{idx + 1}. {event_type}** · `{agent_id}` · {ts}"
                + (f" — {action}" if action else "")
            )
            with st.expander(header, expanded=False):
                meta_cols = st.columns(3)
                meta_cols[0].markdown(f"**event_id**\n\n`{ev.get('event_id', '')}`")
                meta_cols[1].markdown(f"**action**\n\n{action or '—'}")
                meta_cols[2].markdown(f"**outcome**\n\n{outcome or '—'}")

                if ev.get("action_inputs"):
                    st.markdown("**action_inputs**")
                    st.json(ev["action_inputs"])

                st.markdown("**state diff (before → after)**")
                diff = _state_diff(ev.get("state_before", {}), ev.get("state_after", {}))
                if diff.strip():
                    st.code(diff, language="diff")
                else:
                    st.caption("No state change.")

                sb, sa = st.columns(2)
                with sb:
                    st.markdown("**state_before**")
                    st.json(ev.get("state_before", {}) or {})
                with sa:
                    st.markdown("**state_after**")
                    st.json(ev.get("state_after", {}) or {})


# ---------------------------------------------------------------------------
# Tab 2 — Plans
# ---------------------------------------------------------------------------

def _reason_lookup(deviations: List[str], reasons: List[str]) -> Dict[str, str]:
    return {dev: (reasons[i] if i < len(reasons) else "") for i, dev in enumerate(deviations)}


def _render_plan_column(
    title: str,
    actions: List[str],
    deviation_set: set,
    reasons: Dict[str, str],
) -> None:
    st.markdown(f"**{title}**")
    if not actions:
        st.caption("— none —")
        return
    for i, action in enumerate(actions):
        if action in deviation_set:
            reason = reasons.get(action) or "Deviation — no reason recorded."
            st.markdown(
                f"{i + 1}. :red[**{action}**] ⚠",
                help=reason,
            )
        else:
            st.markdown(f"{i + 1}. {action}")


def render_plans_tab(plans: List[Dict[str, Any]]) -> None:
    render_tab_summary("Plans", _summarize_plans(plans))
    if not plans:
        st.info("No plans recorded for this task.")
        return

    # Group by agent so each agent gets its own intended/actual panel.
    by_agent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in plans:
        by_agent[p.get("agent_id") or "unknown"].append(p)

    for agent_id, agent_plans in by_agent.items():
        st.subheader(f"Agent · `{agent_id}`")
        for plan in sorted(agent_plans, key=lambda p: p.get("timestamp", 0)):
            intended = plan.get("intended_actions") or []
            actual = plan.get("actual_actions") or []
            deviations = plan.get("deviations") or []
            reasons = _reason_lookup(deviations, plan.get("deviation_reasons") or [])
            dev_set = set(deviations)

            st.caption(
                f"plan_id `{plan.get('plan_id', '')[:8]}` · "
                f"{_fmt_ts(plan.get('timestamp', 0))} · "
                f"{len(deviations)} deviation(s)"
            )
            c_int, c_act = st.columns(2)
            with c_int:
                _render_plan_column("Intended", intended, dev_set, reasons)
            with c_act:
                _render_plan_column("Actual", actual, dev_set, reasons)

            if deviations:
                st.markdown("**Deviations**")
                for dev in deviations:
                    reason = reasons.get(dev) or "No reason recorded."
                    st.markdown(f"- :red[{dev}]", help=reason)
            st.divider()


# ---------------------------------------------------------------------------
# Tab 3 — Tool Provenance
# ---------------------------------------------------------------------------

def _truncate(value: Any, limit: int = 80) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, default=str)
        except (TypeError, ValueError):
            value = str(value)
    value = value.replace("\n", " ")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def render_tool_provenance_tab(tool_calls: List[Dict[str, Any]]) -> None:
    render_tab_summary("Tool Provenance", _summarize_tools(tool_calls))
    if not tool_calls:
        st.info("No tool calls recorded for this task.")
        return

    calls = sorted(tool_calls, key=lambda t: t.get("timestamp", 0))

    df = pd.DataFrame(
        [
            {
                "#": i + 1,
                "tool": c.get("tool_name", ""),
                "called_by": c.get("called_by", ""),
                "inputs": _truncate(c.get("inputs", {})),
                "outputs": _truncate(c.get("outputs", {})),
                "impact_score": float(c.get("downstream_impact_score") or 0.0),
                "duration_ms": float(c.get("duration_ms") or 0.0),
            }
            for i, c in enumerate(calls)
        ]
    )

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "tool": st.column_config.TextColumn(width="medium"),
            "called_by": st.column_config.TextColumn(width="small"),
            "inputs": st.column_config.TextColumn("inputs (truncated)", width="large"),
            "outputs": st.column_config.TextColumn("outputs (truncated)", width="large"),
            "impact_score": st.column_config.ProgressColumn(
                "impact",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
                help="Counterfactual downstream impact score (0–1).",
            ),
            "duration_ms": st.column_config.NumberColumn(format="%.0f ms", width="small"),
        },
    )

    st.markdown("**Counterfactual re-run**")
    options = [
        f"#{i + 1} · {c.get('tool_name', '?')} · called_by={c.get('called_by', '?')} · "
        f"impact={float(c.get('downstream_impact_score') or 0):.2f}"
        for i, c in enumerate(calls)
    ]
    picked = st.selectbox(
        "Pick a tool call to inspect its counterfactual re-run:",
        options=options,
        index=0,
        key="tool_cf_select",
    )
    if picked is None:
        return
    idx = options.index(picked)
    call = calls[idx]
    cf_id = call.get("counterfactual_run_id") or ""

    meta_a, meta_b, meta_c = st.columns(3)
    meta_a.metric("tool_call_id", (call.get("tool_call_id") or "")[:8])
    meta_b.metric("impact_score", f"{float(call.get('downstream_impact_score') or 0):.2f}")
    meta_c.metric("cf_run_id", (cf_id[:8] if cf_id else "—"))

    st.caption("Original call")
    st.json({"inputs": call.get("inputs", {}), "outputs": call.get("outputs", {})})

    st.caption("Counterfactual re-run")
    if not cf_id:
        st.info("No counterfactual run is linked to this tool call.")
        return
    try:
        cf_record = cached_task_record(cf_id)
    except Exception as e:
        st.warning(f"Could not fetch counterfactual task `{cf_id[:8]}`: {e}")
        return
    st.json(cf_record)


# ---------------------------------------------------------------------------
# Tab 4 — Memory
# ---------------------------------------------------------------------------

def _memory_diff_block(before: Any, after: Any) -> str:
    b_text = json.dumps(before, indent=2, sort_keys=True, default=str).splitlines()
    a_text = json.dumps(after, indent=2, sort_keys=True, default=str).splitlines()
    diff = unified_diff(
        b_text, a_text, fromfile="value_before", tofile="value_after", lineterm=""
    )
    return "\n".join(diff)


def _find_event(trajectory: List[Dict[str, Any]], event_id: str) -> Optional[Dict[str, Any]]:
    if not event_id:
        return None
    for ev in trajectory:
        if ev.get("event_id") == event_id:
            return ev
    return None


def render_memory_tab(
    memory_diffs: List[Dict[str, Any]],
    trajectory: List[Dict[str, Any]],
) -> None:
    render_tab_summary("Memory", _summarize_memory(memory_diffs))
    if not memory_diffs:
        st.info("No memory writes recorded for this task.")
        return

    writes = [m for m in memory_diffs if (m.get("operation") or "").lower() == "write"]
    if not writes:
        st.info("Memory activity recorded, but no writes.")
        return

    by_agent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in writes:
        by_agent[m.get("agent_id") or "unknown"].append(m)

    for agent_id, agent_writes in by_agent.items():
        agent_writes.sort(key=lambda m: m.get("timestamp", 0))
        with st.expander(f"Agent · `{agent_id}` · {len(agent_writes)} write(s)", expanded=False):
            for i, m in enumerate(agent_writes):
                diff_id = m.get("diff_id", f"d{i}")
                key = m.get("key", "?")
                st.markdown(
                    f"**{i + 1}. `{key}`** · "
                    f"{_fmt_ts(m.get('timestamp', 0))} · "
                    f"diff_id `{diff_id[:8]}`"
                )

                d_before, d_after = st.columns(2)
                with d_before:
                    st.caption("value_before")
                    st.code(
                        json.dumps(m.get("value_before"), indent=2, default=str),
                        language="json",
                    )
                with d_after:
                    st.caption("value_after")
                    st.code(
                        json.dumps(m.get("value_after"), indent=2, default=str),
                        language="json",
                    )

                diff_text = _memory_diff_block(m.get("value_before"), m.get("value_after"))
                if diff_text.strip():
                    st.code(diff_text, language="diff")

                # Pointer to triggering event. Streamlit forbids nesting
                # st.expander inside another expander, so we simulate the
                # "clickable expander" with a toggle that reveals the event.
                trig_id = m.get("triggered_by_event_id") or ""
                if trig_id:
                    if st.toggle(
                        f"🔗 Triggered by event `{trig_id[:8]}`",
                        key=f"mem_trig_{diff_id}",
                    ):
                        event = _find_event(trajectory, trig_id)
                        if event is None:
                            st.caption("Triggering event not found in trajectory.")
                        else:
                            st.json(event)
                else:
                    st.caption("No triggering event linked.")

                st.divider()


# ---------------------------------------------------------------------------
# Shared graph helpers (Tabs 5 & 6)
# ---------------------------------------------------------------------------

def _pyvis_html(net: Any) -> Optional[str]:
    """Return pyvis graph HTML as a string across pyvis versions."""
    try:
        return net.generate_html(notebook=False)
    except TypeError:
        try:
            return net.generate_html()
        except Exception:
            return None
    except Exception:
        return None


def _import_pyvis() -> Any:
    try:
        from pyvis.network import Network
    except ImportError:
        st.error(
            "pyvis is required for this tab. Install with `pip install pyvis==0.3.2` "
            "and restart Streamlit."
        )
        return None
    return Network


# ---------------------------------------------------------------------------
# Tab 5 — Communication
# ---------------------------------------------------------------------------

def render_communication_tab(messages: List[Dict[str, Any]]) -> None:
    render_tab_summary("Communication", _summarize_messages(messages))
    if not messages:
        st.info("No inter-agent messages recorded for this task.")
        return

    Network = _import_pyvis()
    if Network is None:
        return

    net = Network(
        height="520px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#ffffff",
        font_color="#222222",
    )
    net.force_atlas_2based()

    agents: List[str] = []
    for m in messages:
        for a in (m.get("sender"), m.get("receiver")):
            a = a or "unknown"
            if a not in agents:
                agents.append(a)

    agent_colors = {a: AGENT_PALETTE[i % len(AGENT_PALETTE)] for i, a in enumerate(agents)}
    for a in agents:
        net.add_node(
            a,
            label=a,
            shape="dot",
            size=28,
            color=agent_colors[a],
            title=f"agent: {a}",
        )

    for i, m in enumerate(messages):
        sender = m.get("sender") or "unknown"
        receiver = m.get("receiver") or "unknown"
        acted = bool(m.get("acted_upon"))
        # Spec: 1.0 if true, 0.3 if false. Multiply for visible stroke width.
        thickness = 1.0 if acted else 0.3
        width_px = 1.0 + thickness * 5.0
        edge_color = "#2E86AB" if acted else "#bdbdbd"
        label_bits = [m.get("message_type") or "message"]
        if m.get("behavior_change_description"):
            label_bits.append(m["behavior_change_description"])
        net.add_edge(
            sender,
            receiver,
            value=width_px,
            width=width_px,
            color=edge_color,
            title="<br>".join(label_bits) + f"<br>acted_upon={acted}",
            arrows="to",
        )

    html = _pyvis_html(net)
    if html is None:
        st.error("Failed to render pyvis graph.")
    else:
        components.html(html, height=540, scrolling=False)

    st.caption(
        "pyvis does not forward click events to Streamlit — use the selectors below "
        "to open a details pane."
    )

    st.divider()
    st.markdown("**Details**")
    col_n, col_e = st.columns(2)

    with col_n:
        node_pick = st.selectbox(
            "Inspect an agent (node)",
            options=["—"] + agents,
            index=0,
            key="comm_node_select",
        )
        if node_pick and node_pick != "—":
            sent = [m for m in messages if (m.get("sender") or "") == node_pick]
            received = [m for m in messages if (m.get("receiver") or "") == node_pick]
            st.metric("sent", len(sent))
            st.metric("received", len(received))
            if sent or received:
                st.caption("Recent messages involving this agent")
                st.json([m for m in (sent + received)][:10])

    with col_e:
        edge_labels = [
            f"#{i + 1} · {m.get('sender', '?')} → {m.get('receiver', '?')} · "
            f"{m.get('message_type', 'message')}"
            for i, m in enumerate(messages)
        ]
        edge_pick = st.selectbox(
            "Inspect a message (edge)",
            options=["—"] + edge_labels,
            index=0,
            key="comm_edge_select",
        )
        if edge_pick and edge_pick != "—":
            idx = edge_labels.index(edge_pick)
            msg = messages[idx]
            st.metric("acted_upon", "yes" if msg.get("acted_upon") else "no")
            if msg.get("behavior_change_description"):
                st.markdown(f"**Behavior change:** {msg['behavior_change_description']}")
            st.json(msg)


# ---------------------------------------------------------------------------
# Tab 6 — Causality
# ---------------------------------------------------------------------------

def _event_short_label(ev: Optional[Dict[str, Any]], fallback_id: str) -> str:
    if not ev:
        return fallback_id[:8]
    et = ev.get("event_type") or "event"
    agent = ev.get("agent_id") or "?"
    return f"{et}\n{agent}\n{fallback_id[:8]}"


def render_causality_tab(
    causal_graph: Dict[str, Any],
    trajectory: List[Dict[str, Any]],
    report: Optional[Dict[str, Any]],
) -> None:
    render_tab_summary("Causality", _summarize_causality(causal_graph, report))
    nodes: List[str] = (causal_graph or {}).get("nodes") or []
    edges: List[Dict[str, Any]] = (causal_graph or {}).get("edges") or []
    if not nodes and not edges:
        st.info("No causal graph recorded for this task.")
        return

    Network = _import_pyvis()
    if Network is None:
        return

    root = ((report or {}).get("root_cause_event_id") or "").strip()
    ev_map = {e.get("event_id"): e for e in trajectory}

    # Union of explicit node list and any endpoints referenced by edges.
    node_set: List[str] = list(nodes)
    for e in edges:
        for nid in (e.get("cause_event_id"), e.get("effect_event_id")):
            if nid and nid not in node_set:
                node_set.append(nid)

    net = Network(
        height="560px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#ffffff",
        font_color="#222222",
    )
    net.barnes_hut()

    for nid in node_set:
        ev = ev_map.get(nid)
        is_root = (nid == root)
        label = _event_short_label(ev, nid)
        title_lines = [
            f"event_id: {nid}",
            f"type: {(ev or {}).get('event_type', '?')}",
            f"agent: {(ev or {}).get('agent_id', '?')}",
            f"action: {(ev or {}).get('action', '')}",
        ]
        if is_root:
            title_lines.append("ROOT CAUSE")
        net.add_node(
            nid,
            label=label,
            title="<br>".join(title_lines),
            color={"background": "#E63946", "border": "#8A0A1B"} if is_root else "#2E86AB",
            size=38 if is_root else 22,
            borderWidth=3 if is_root else 1,
            shape="dot",
        )

    for e in edges:
        strength = float(e.get("causal_strength") or 0.0)
        strength = max(0.0, min(1.0, strength))
        alpha = max(0.15, strength)  # keep weak edges visible
        color = f"rgba(101, 67, 181, {alpha:.2f})"  # purple, intensity = strength
        width_px = 1.0 + strength * 5.0
        net.add_edge(
            e.get("cause_event_id"),
            e.get("effect_event_id"),
            value=width_px,
            width=width_px,
            color=color,
            title=(
                f"type: {e.get('causal_type', '?')}<br>"
                f"strength: {strength:.2f}"
            ),
            arrows="to",
        )

    html = _pyvis_html(net)
    if html is None:
        st.error("Failed to render pyvis graph.")
    else:
        components.html(html, height=580, scrolling=False)

    if root:
        st.caption(f"🔴 Root-cause node highlighted: `{root[:8]}`")

    st.divider()
    st.markdown("**Causal edges**")

    rows = []
    for e in edges:
        cause_id = e.get("cause_event_id", "") or ""
        effect_id = e.get("effect_event_id", "") or ""
        cause_ev = ev_map.get(cause_id)
        effect_ev = ev_map.get(effect_id)
        rows.append(
            {
                "cause_id": cause_id[:8],
                "cause": (cause_ev or {}).get("event_type", "?")
                + " · "
                + ((cause_ev or {}).get("agent_id") or "?"),
                "effect_id": effect_id[:8],
                "effect": (effect_ev or {}).get("event_type", "?")
                + " · "
                + ((effect_ev or {}).get("agent_id") or "?"),
                "type": e.get("causal_type", ""),
                "strength": float(e.get("causal_strength") or 0.0),
            }
        )
    df = pd.DataFrame(rows)

    query = st.text_input(
        "🔍 Search (matches cause, effect, type, or id)",
        key="causal_filter",
        placeholder="e.g. direct, orchestrator, tool_call…",
    )
    if query:
        q = query.lower()
        mask = df.apply(lambda r: q in " ".join(str(v) for v in r.values).lower(), axis=1)
        df = df[mask]

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "cause_id": st.column_config.TextColumn("cause id", width="small"),
            "effect_id": st.column_config.TextColumn("effect id", width="small"),
            "strength": st.column_config.ProgressColumn(
                "strength", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )


# ---------------------------------------------------------------------------
# Tab 7 — Accountability
# ---------------------------------------------------------------------------

def _event_has_tool_call(ev: Dict[str, Any], tool_call_id: str) -> bool:
    if not tool_call_id:
        return False
    if ev.get("event_type") != "tool_call":
        return False
    inputs = ev.get("action_inputs") or {}
    if inputs.get("tool_call_id") == tool_call_id:
        return True
    outcome = ev.get("outcome") or ""
    return tool_call_id in outcome


def render_accountability_tab(
    report: Optional[Dict[str, Any]],
    trajectory: List[Dict[str, Any]],
    tool_calls: List[Dict[str, Any]],
) -> None:
    render_tab_summary("Accountability", _summarize_accountability(report))
    if not report:
        st.info("No accountability report recorded for this task.")
        return

    explanation = report.get("one_line_explanation") or "—"
    correct = report.get("outcome_correct")
    accent = "#06A77D" if correct else ("#E63946" if correct is False else "#2E86AB")

    st.markdown(
        f"""
        <div style="padding:28px 32px;background:#f4f6f8;border-left:8px solid {accent};
                    border-radius:8px;margin-bottom:20px;">
          <div style="font-size:0.8rem;letter-spacing:0.08em;text-transform:uppercase;
                      color:#666;margin-bottom:10px;">One-line explanation</div>
          <div style="font-size:1.7rem;font-weight:600;line-height:1.35;color:#1b1b1b;">
            {explanation}
          </div>
          <div style="margin-top:14px;font-size:0.95rem;color:#444;">
            Final outcome: <b>{report.get("final_outcome", "—")}</b>
            &nbsp;·&nbsp; Correct:
            <b style="color:{accent};">{"✅" if correct else ("❌" if correct is False else "—")}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Responsibility bar chart (horizontal).
    scores: Dict[str, float] = report.get("agent_responsibility_scores") or {}
    if scores:
        st.markdown("**Agent responsibility scores**")
        df_scores = (
            pd.DataFrame({"responsibility": scores})
            .sort_values("responsibility", ascending=True)
        )
        try:
            st.bar_chart(df_scores, horizontal=True)
        except TypeError:
            # Older Streamlit without horizontal= support.
            st.bar_chart(df_scores)
    else:
        st.caption("No responsibility scores recorded.")

    st.divider()

    # Causal chain as numbered, clickable expanders.
    chain: List[str] = report.get("causal_chain") or []
    root = report.get("root_cause_event_id") or ""
    impactful_tool = report.get("most_impactful_tool_call_id") or ""

    st.markdown("**Causal chain**")
    if not chain:
        st.caption("No causal chain recorded.")
    else:
        ev_map = {e.get("event_id"): e for e in trajectory}
        for i, eid in enumerate(chain):
            ev = ev_map.get(eid) or {"event_id": eid}
            is_root = (eid == root)
            is_impactful = _event_has_tool_call(ev, impactful_tool)

            badges = []
            if is_root:
                badges.append("🔴 root cause")
            if is_impactful:
                badges.append("⭐ most-impactful tool call")
            badge_str = (" · " + " · ".join(badges)) if badges else ""

            header = (
                f"{i + 1}. {ev.get('event_type', '?')} · "
                f"`{(ev.get('agent_id') or '?')}` · "
                f"{_fmt_ts(ev.get('timestamp', 0)) if ev.get('timestamp') else '—'}"
                f"{badge_str}"
            )
            with st.expander(header, expanded=is_root):
                if is_root:
                    st.markdown(":red[**Flagged as root cause of the outcome.**]")
                if is_impactful:
                    tool_match = next(
                        (t for t in tool_calls if t.get("tool_call_id") == impactful_tool),
                        None,
                    )
                    if tool_match:
                        st.markdown("**Most-impactful tool call**")
                        st.json(tool_match)
                st.markdown("**Event**")
                st.json(ev)

    st.divider()

    # Supporting pointers.
    m1, m2, m3 = st.columns(3)
    m1.metric("Root-cause event", (root[:8] if root else "—"))
    m2.metric("Most-impactful tool call", (impactful_tool[:8] if impactful_tool else "—"))
    m3.metric(
        "Most-influential message",
        ((report.get("most_influential_message_id") or "")[:8] or "—"),
    )

    if report.get("plan_deviation_summary"):
        st.markdown("**Plan deviation summary**")
        st.write(report["plan_deviation_summary"])

    critical_diffs = report.get("critical_memory_diffs") or []
    if critical_diffs:
        st.markdown("**Critical memory diffs**")
        st.write([d[:8] for d in critical_diffs])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="AgentXAI", layout="wide")
    inject_css()

    # Default view — per-case summary lands first; user clicks through to
    # the 7 pillar tabs for that same task.
    if "view" not in st.session_state:
        st.session_state["view"] = "landing"

    task_id = render_sidebar()

    if not task_id:
        st.markdown(
            """
            <div class="xai-hero">
              <div class="xai-hero-eyebrow">AgentXAI · Multi-agent explainability</div>
              <h1 class="xai-hero-title">Pick a task to see its full XAI trace.</h1>
              <div class="xai-hero-meta">
                <span class="xai-pill">Each task lands on a per-case summary</span>
                <span class="xai-pill">Then drill into the 7 pillar views</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "Use the **Task** dropdown in the sidebar to select an existing run, "
            "or expand **▶ Run new task** to submit a new MedQA record."
        )
        return

    try:
        record = cached_task_record(task_id)
    except Exception as e:
        st.error(f"Could not fetch /tasks/{task_id}: {e}")
        return

    if st.session_state["view"] == "landing":
        render_case_overview(record)
        return

    # Pillar view — slim case strip up top so the user always knows which
    # case they're looking at.
    render_case_strip(record)

    xai = record.get("xai_data") or {}
    trajectory = xai.get("trajectory", []) or []
    plans = xai.get("plans", []) or []
    tool_calls = xai.get("tool_calls", []) or []
    memory_diffs = xai.get("memory_diffs", []) or []
    messages = xai.get("messages", []) or []
    causal_graph = xai.get("causal_graph", {}) or {}
    accountability_report = xai.get("accountability_report")

    tabs = st.tabs(TAB_LABELS)

    with tabs[0]:
        render_trajectory_tab(trajectory)
    with tabs[1]:
        render_plans_tab(plans)
    with tabs[2]:
        render_tool_provenance_tab(tool_calls)
    with tabs[3]:
        render_memory_tab(memory_diffs, trajectory)
    with tabs[4]:
        render_communication_tab(messages)
    with tabs[5]:
        render_causality_tab(causal_graph, trajectory, accountability_report)
    with tabs[6]:
        render_accountability_tab(accountability_report, trajectory, tool_calls)


if __name__ == "__main__":
    main()
