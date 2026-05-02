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


def _api_headers() -> Dict[str, str]:
    """
    Build outbound request headers.

    Read ``AGENTXAI_API_TOKEN`` at call time (not module load time) so a
    running Streamlit session picks up a freshly exported token without
    a server restart. When unset, no Authorization header is sent and
    the API treats the call as un-authed (matching its local-dev default).
    """
    token = (os.environ.get("AGENTXAI_API_TOKEN") or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}

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


# Per-tab plain-English explainers. Prepended to each _summarize_*
# headline so the blue summary box reads like a "what this tab shows"
# sentence followed by the count summary, instead of a stat dump.
_PILLAR_EXPLAINERS: Dict[str, str] = {
    "trajectory": (
        "This tab shows the full step-by-step execution trace. The case "
        "moves from the Orchestrator through the specialists to the "
        "Synthesizer, with every action timestamped."
    ),
    "plans": (
        "This tab checks whether each agent followed its intended plan, "
        "and surfaces any deviation between intended and actual actions."
    ),
    "tools": (
        "This tab explains which tools were called and which one affected "
        "the answer most. <code>pubmed_search</code> is a "
        "<i>local textbook FAISS search</i>, not the real PubMed API — "
        "the historical name is preserved on stored records."
    ),
    "memory": (
        "This tab shows what each agent wrote to memory and (when "
        "<code>memory_usage</code> attribution is available) whether the "
        "Synthesizer actually cited it in the final rationale."
    ),
    "messages": (
        "This tab shows messages between agents and whether downstream "
        "agents acted on them — an unacted-upon message is a strong hint "
        "that its sender's contribution was ignored."
    ),
    "causality": (
        "This tab shows the causal graph connecting earlier events to the "
        "final answer. Stronger edges indicate larger estimated influence "
        "via counterfactual perturbations."
    ),
    "accountability": (
        "This tab combines all XAI signals into a final accountability "
        "report. Responsibility is a <b>composite</b> of counterfactual "
        "impact, tool impact, acted-upon messages, used memory, "
        "self-reported usefulness, and causal centrality — weighted by "
        "question type. See <code>XAIScoringConfig</code> for the knobs."
    ),
}


def _prepend_explainer(tab_key: str, headline: str) -> str:
    """Return the explainer paragraph + the original count headline."""
    explainer = _PILLAR_EXPLAINERS.get(tab_key)
    if not explainer:
        return headline
    return (
        f"<div style='margin-bottom:8px;color:#222;'>{explainer}</div>"
        f"{headline}"
    )


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
    return _prepend_explainer("trajectory", headline), bullets


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
    return _prepend_explainer("plans", headline), bullets


# Tools whose internal `tool_name` is preserved for backward compatibility
# but whose actual implementation is something different from what the name
# suggests. The dashboard surfaces the truth in user-facing strings while
# leaving stored records (`tool_use_events.tool_name`) untouched.
#
# `pubmed_search` is currently a local FAISS retrieval over 18 medical
# textbooks — see `agentxai/tools/pubmed_search.py` for why the name is
# kept and how to swap in real PubMed.
_TOOL_DISPLAY_OVERRIDES: Dict[str, str] = {
    "pubmed_search": "pubmed_search (local textbook FAISS)",
}


def _tool_display_name(tool_name: Optional[str]) -> str:
    """User-facing tool label, with overrides for misleading historical names."""
    if not tool_name:
        return "?"
    return _TOOL_DISPLAY_OVERRIDES.get(tool_name, tool_name)


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
    top_name = _tool_display_name(top.get("tool_name"))
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
        name = _tool_display_name(c.get("tool_name"))
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
    return _prepend_explainer("tools", headline), bullets


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
    return _prepend_explainer("memory", headline), bullets


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
    return _prepend_explainer("messages", headline), bullets


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
    return _prepend_explainer("causality", headline), bullets


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
    # Surface the root-cause reason as a bullet so the user can see *why*
    # the selector picked that event without opening the JSON expander.
    root_reason = report.get("root_cause_reason") or ""
    if root_reason:
        bullets.append(f"Root-cause reason: <i>{root_reason}</i>")
    return _prepend_explainer("accountability", headline), bullets


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


def _extract_top_evidence(xai: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Pull Specialist B's most-recent ``top_evidence`` value from the task's
    memory diffs. Returns ``[]`` when no Specialist B evidence write has
    been recorded (older runs, or runs where SpecialistB short-circuited).
    """
    from agentxai.xai.evidence_attribution import (
        latest_top_evidence_from_memory_diffs,
    )
    diffs = (xai or {}).get("memory_diffs") or []
    return latest_top_evidence_from_memory_diffs(diffs)


def _render_evidence_cards(
    top_evidence: List[Dict[str, Any]],
    used_ids: List[str],
) -> None:
    """
    Render one expandable card per retrieved evidence doc.

    Each card header shows: doc_id (short), source_file, FAISS score, and
    a "✓ used in rationale" badge when applicable. Clicking the expander
    reveals the full snippet. Renders nothing when there's no evidence —
    older records and short-circuited runs don't get a stray empty header.
    """
    if not top_evidence:
        return

    used_set = {str(x) for x in (used_ids or []) if x}
    used_count = sum(
        1 for ev in top_evidence
        if isinstance(ev, dict) and str(ev.get("doc_id") or "") in used_set
    )

    st.markdown(
        '<div class="xai-card-eyebrow" style="margin-top:14px;">'
        f'Retrieved evidence · {len(top_evidence)} doc(s) · '
        f'{used_count} cited in rationale'
        '</div>',
        unsafe_allow_html=True,
    )

    for i, ev in enumerate(top_evidence):
        if not isinstance(ev, dict):
            continue
        doc_id = str(ev.get("doc_id") or "").strip() or f"doc-{i + 1}"
        source = str(ev.get("source_file") or "?")
        try:
            score = float(ev.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        snippet = str(ev.get("snippet") or ev.get("text") or "").strip()
        is_used = doc_id in used_set
        used_badge = "✓ used" if is_used else "·"
        score_str = f"{score:.2f}" if score else "—"
        # Truncate the doc_id so long FAISS chunk names don't dominate.
        short_id = doc_id if len(doc_id) <= 32 else doc_id[:29] + "…"
        header = (
            f"{used_badge}  `{short_id}`  ·  {source}  ·  score {score_str}"
        )
        with st.expander(header, expanded=False):
            if snippet:
                st.write(snippet)
            else:
                st.caption("_(no snippet recorded)_")
            cols = st.columns(3)
            cols[0].metric("doc_id", short_id)
            cols[1].metric("source_file", source)
            cols[2].metric(
                "score", score_str,
                help="FAISS cosine similarity to the patient case.",
            )
            if is_used:
                st.caption(
                    "✓ This doc's content appears in the Synthesizer's "
                    "rationale (explicit `supporting_evidence_ids` or "
                    "heuristic substring match)."
                )
            else:
                st.caption(
                    "Not cited in the rationale — included here as a "
                    "high-similarity retrieval result."
                )


# Display labels + tooltips for confidence_factors. Order matches
# `agentxai.xai.confidence_factors.FACTOR_KEYS`. Factor names are kept
# verbatim (snake_case) in the dashboard so they line up with the JSON
# field surfaced via the API and the heuristic module.
_CONFIDENCE_FACTOR_LABELS: Dict[str, str] = {
    "retrieval_relevance":
        "Mean similarity of Specialist B's retrieved evidence (FAISS).",
    "option_match_strength":
        "How cleanly the chosen option matches the listed options + per-option verdict.",
    "specialist_agreement":
        "Fraction of specialists whose memory mentions the predicted diagnosis.",
    "evidence_coverage":
        "Number of supporting evidence ids cited (target: 3+).",
    "contradiction_penalty":
        "Fraction of options marked 'partial' — competing candidates.",
}


def _render_confidence_factors_panel(
    *,
    confidence: Any,
    factors: Dict[str, Any],
) -> None:
    """
    Render the heuristic confidence breakdown next to the prediction.

    Skips silently when `factors` is empty — older records produced before
    confidence_factors existed don't carry the field, and the panel
    shouldn't appear with all-zero rows in that case.

    The panel intentionally calls confidence "heuristic" — these factors
    do NOT add up to the headline confidence; they're soft observable
    signals showing what *could* be driving it. The headline is the LLM's
    self-report and is not clinically calibrated.
    """
    if not isinstance(factors, dict) or not factors:
        return

    conf_str = (
        f"{float(confidence):.2f}"
        if isinstance(confidence, (int, float)) else "—"
    )
    rows = []
    for key, label in _CONFIDENCE_FACTOR_LABELS.items():
        if key not in factors:
            continue
        try:
            val = max(0.0, min(1.0, float(factors[key])))
        except (TypeError, ValueError):
            continue
        pct = val * 100
        rows.append(
            '<div style="margin:6px 0;">'
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:0.85rem;">'
            f'<span title="{label}"><b>{key}</b></span>'
            f'<span style="color:#444;">{val:.2f}</span>'
            '</div>'
            '<div style="height:6px;background:#eee;border-radius:3px;'
            'overflow:hidden;">'
            f'<div style="height:100%;width:{pct:.1f}%;background:#2E86AB;"></div>'
            '</div>'
            '</div>'
        )
    if not rows:
        return

    st.markdown(
        f'<div class="xai-card" style="margin-top:14px;padding:14px 16px;">'
        f'<div class="xai-card-eyebrow">Confidence breakdown · headline {conf_str}</div>'
        + "".join(rows)
        + '<div style="margin-top:8px;font-size:0.75rem;color:#888;">'
        '⚠ Heuristic only — these factors decompose the LLM\'s self-reported '
        'confidence into observable signals. They are <b>not clinically '
        'calibrated</b> and do not represent a probability of correctness.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Summary-page helpers (case story, what mattered, evidence, faithfulness)
# ---------------------------------------------------------------------------
#
# Each function takes the loose JSON-shaped `record` dict the API returns and
# produces a small, render-ready string or list. They're factored out of
# render_case_overview so:
#   * each card's prose is testable in isolation,
#   * an old record with missing fields degrades to a graceful fallback line
#     instead of crashing or showing "—" everywhere,
#   * the Summary page reads top-to-bottom as a story, not a stat dump.

# Aggregator action labels used by `_is_aggregator_event` and the
# faithfulness snapshot. Mirrors the canonical list in
# `agentxai.xai.accountability` and `agentxai.ui.faithfulness_checks`.
_SUMMARY_AGGREGATOR_ACTIONS = frozenset({
    "read_specialist_memories", "handoff_to_synthesizer",
    "decompose_case", "dispatch_specialists",
    "aggregate_findings", "compile_results",
})
_SUMMARY_AGGREGATOR_PREFIXES = ("route_to_", "handoff_to_", "dispatch_")


def _is_aggregator_event(event: Optional[Dict[str, Any]]) -> bool:
    """True iff `event` is a routing / aggregation step, not a real decision."""
    if not isinstance(event, dict):
        return False
    action = str(event.get("action") or "").strip().lower()
    if action in _SUMMARY_AGGREGATOR_ACTIONS:
        return True
    if any(action.startswith(p) for p in _SUMMARY_AGGREGATOR_PREFIXES):
        return True
    et = str(event.get("event_type") or "").strip().lower()
    return et in {"plan", "routing"}


# Substring the selector appends to `root_cause_reason` when every
# ancestor of the terminal was an aggregator and it had to fall back.
# Surfacing this distinct case in the UI is the whole point of the
# staleness check below.
_FALLBACK_MARKER_SUBSTR = "no non-aggregator ancestor"


def _is_stale_accountability_report(report: Optional[Dict[str, Any]]) -> bool:
    """
    True if the report was generated by code that predates the new
    selector + memory_usage attribution, OR if the selector ran but had
    to fall back to an aggregator root cause (i.e., no causal edges
    from specialists to the terminal — usually because the run predates
    `traced_action`-linked memory writes).

    The Summary and Accountability surfaces use this to decide whether
    to show the "Re-run the task to refresh root-cause attribution"
    banner. The check is deliberately conservative: a non-stale report
    has populated `root_cause_reason` AND a non-aggregator root cause.
    """
    if not isinstance(report, dict) or not report:
        return False  # No report to be stale about.

    # Pre-selector records have no `root_cause_reason` field populated.
    if not (report.get("root_cause_reason") or "").strip():
        return True

    # Selector ran but fell back to an aggregator — surfaced in the
    # reason text via the marker the selector appends. This is a
    # post-fix record but the run itself is degenerate.
    reason = (report.get("root_cause_reason") or "").lower()
    if _FALLBACK_MARKER_SUBSTR in reason:
        return True

    # Belt-and-suspenders: the report's `memory_usage` field is empty
    # AND the underlying record has memory writes — means the
    # attribution pass never ran for this report. (Empty `memory_usage`
    # is fine when the agents genuinely wrote nothing — covered by the
    # report-only check above falling through.)
    return False


def _staleness_message(report: Optional[Dict[str, Any]]) -> str:
    """
    Return the user-facing banner text for a stale report, or "" if not
    stale. Distinguishes the two staleness modes so the message points
    at the right remediation:

      * Pre-selector records get the canonical "Re-run the task to
        refresh root-cause attribution" message.
      * Post-selector aggregator-fallback records get a more nuanced
        message naming the underlying cause.
    """
    if not _is_stale_accountability_report(report):
        return ""
    reason = (report.get("root_cause_reason") or "").lower()
    if not reason:
        return (
            "This task was generated with an older accountability method. "
            "Re-run the task to refresh root-cause attribution."
        )
    if _FALLBACK_MARKER_SUBSTR in reason:
        return (
            "Selector fell back to an aggregator event because no "
            "non-aggregator ancestor was found in the causal graph — "
            "usually because this run predates `traced_action`-linked "
            "memory writes. Re-run the task to refresh attribution."
        )
    return (
        "This task's accountability report looks stale. "
        "Re-run the task to refresh attribution."
    )


def _render_staleness_banner(report: Optional[Dict[str, Any]]) -> None:
    """
    Render an in-place yellow warning banner when the report is stale.
    Renders nothing for fresh reports.
    """
    msg = _staleness_message(report)
    if not msg:
        return
    st.markdown(
        '<div style="margin:14px 0;padding:12px 16px;border-radius:6px;'
        'background:#fff7e6;border-left:4px solid #F4A261;color:#7a4f00;">'
        f'⚠ <b>{msg}</b>'
        '</div>',
        unsafe_allow_html=True,
    )


def _top_responsible_agent(report: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return the single highest-responsibility agent_id, or None."""
    if not report:
        return None
    scores = report.get("agent_responsibility_scores") or {}
    if not scores:
        return None
    return max(scores.items(), key=lambda kv: float(kv[1]))[0]


def _get_supporting_evidence(
    record: Dict[str, Any], top_n: int = 3,
) -> List[Dict[str, Any]]:
    """
    Return up to `top_n` supporting evidence dicts the Summary should
    surface. Prefers items the rationale *used* (per
    `system_output.supporting_evidence_ids`); fills the rest from the
    highest-FAISS-score retrieved docs to give reviewers context even
    when the LLM under-cited.
    """
    xai = record.get("xai_data") or {}
    system_output = record.get("system_output") or {}
    top_evidence = _extract_top_evidence(xai)
    if not top_evidence:
        return []

    used_ids = {
        str(x) for x in (system_output.get("supporting_evidence_ids") or [])
        if x
    }
    used: List[Dict[str, Any]] = []
    others: List[Dict[str, Any]] = []
    for ev in top_evidence:
        if not isinstance(ev, dict):
            continue
        annotated = dict(ev)
        annotated["used_in_rationale"] = (
            str(ev.get("doc_id") or "") in used_ids
        )
        if annotated["used_in_rationale"]:
            used.append(annotated)
        else:
            others.append(annotated)
    others.sort(
        key=lambda e: float(e.get("score") or 0.0), reverse=True,
    )
    return (used + others)[:top_n]


def _build_final_answer_sentence(record: Dict[str, Any]) -> str:
    """
    Produce the lead-line sentence for the Final Answer card.

    Example: "The system answered E — 'HIV-1/HIV-2 antibody differentiation
    immunoassay' — and this matched the ground truth. Confidence was 0.90.
    The case was classified as a screening_or_test question."
    """
    system_output = record.get("system_output") or {}
    ground_truth = record.get("ground_truth") or {}
    input_data = record.get("input") or {}

    predicted_letter = (
        system_output.get("predicted_letter")
        or ground_truth.get("correct_answer")
        or ""
    )
    predicted_text = (
        system_output.get("predicted_text")
        or system_output.get("final_diagnosis")
        or ""
    )
    correct = system_output.get("correct")
    confidence = system_output.get("confidence")
    qtype = input_data.get("question_type") or "unknown"
    correct_letter = ground_truth.get("correct_answer") or ""

    parts: List[str] = []
    if predicted_letter and predicted_text:
        parts.append(f"The system answered {predicted_letter} — '{predicted_text}'")
    elif predicted_text:
        parts.append(f"The system answered '{predicted_text}'")
    else:
        parts.append("The system did not produce a final answer for this case")

    if correct is True:
        parts.append("and this matched the ground truth")
    elif correct is False:
        truth_str = (
            f" (correct answer was {correct_letter})"
            if correct_letter else ""
        )
        parts.append(f"but this did NOT match the ground truth{truth_str}")
    else:
        parts.append("but no ground-truth label was recorded")

    sentence = ", ".join(parts) + "."
    if isinstance(confidence, (int, float)):
        sentence += f" Confidence was {float(confidence):.2f}."
    if qtype and qtype != "unknown":
        sentence += f" The case was classified as a {qtype} question."
    return sentence


def _build_pipeline_story_bullets(record: Dict[str, Any]) -> List[str]:
    """
    Describe the agent flow as a short bullet list, dynamically detecting
    which stages actually ran for this record.

    Falls back to a static four-step story when the trajectory is too sparse
    to infer the flow from observed events.
    """
    xai = record.get("xai_data") or {}
    trajectory = xai.get("trajectory") or []
    tool_calls = xai.get("tool_calls") or []

    agents_seen = {ev.get("agent_id") for ev in trajectory if ev.get("agent_id")}

    bullets: List[str] = []

    if "orchestrator" in agents_seen:
        bullets.append(
            "<b>Orchestrator</b> decomposed the case and routed it to the specialists."
        )
    if "specialist_a" in agents_seen:
        bullets.append(
            "<b>Specialist A</b> performed symptom extraction, "
            "condition lookup, and severity scoring."
        )
    if "specialist_b" in agents_seen:
        # If the search actually ran, name the tool honestly.
        ran_search = any(
            (t.get("tool_name") or "").lower() == "pubmed_search"
            for t in tool_calls
        )
        if ran_search:
            bullets.append(
                "<b>Specialist B</b> retrieved medical evidence using "
                "<i>local textbook FAISS search</i> and matched candidate "
                "conditions against guideline stubs."
            )
        else:
            bullets.append(
                "<b>Specialist B</b> ran candidate-condition retrieval + guideline lookup."
            )
    if "synthesizer" in agents_seen:
        bullets.append(
            "<b>Synthesizer</b> read both specialists' memories and produced the final answer."
        )
    bullets.append(
        "The <b>XAI runtime</b> logged trajectory, plans, tool provenance, "
        "memory diffs, messages, the causal DAG, and the accountability report."
    )
    if not bullets:
        bullets = [
            "Pipeline trace is sparse for this record — cannot reconstruct "
            "the agent flow in detail."
        ]
    return bullets


def _build_what_mattered_paragraph(record: Dict[str, Any]) -> str:
    """
    Plain-English narrative of which signals drove the outcome.

    Consults the accountability report for the top agent, the most-impactful
    tool, the most-influential message, and the root-cause event. Adds an
    "Specialist A contributed little" sentence when its share is below a
    small threshold and it has no observable signals.
    """
    xai = record.get("xai_data") or {}
    report = xai.get("accountability_report") or {}
    if not report:
        return (
            "No accountability report recorded for this task — the run may "
            "not have completed the full XAI pipeline."
        )

    scores = report.get("agent_responsibility_scores") or {}
    tool_calls = xai.get("tool_calls") or []
    messages = xai.get("messages") or []
    trajectory = xai.get("trajectory") or []
    ev_map = {e.get("event_id"): e for e in trajectory}

    top_agent = _top_responsible_agent(report)
    top_score = float(scores.get(top_agent, 0.0)) if top_agent else 0.0

    # Tool the report flagged as most impactful (resolved to its name).
    impactful_tool_id = report.get("most_impactful_tool_call_id") or ""
    tool_name, tool_caller, tool_score = "", "", 0.0
    if impactful_tool_id:
        tool = next(
            (t for t in tool_calls if t.get("tool_call_id") == impactful_tool_id),
            None,
        )
        if tool:
            tool_name = _tool_display_name(tool.get("tool_name"))
            tool_caller = tool.get("called_by") or ""
            tool_score = float(tool.get("downstream_impact_score") or 0.0)

    # Most-influential message (resolved to sender → receiver).
    msg_id = report.get("most_influential_message_id") or ""
    msg_pair = ""
    msg_acted = False
    if msg_id:
        msg = next((m for m in messages if m.get("message_id") == msg_id), None)
        if msg:
            msg_pair = f"{msg.get('sender', '?')} → {msg.get('receiver', '?')}"
            msg_acted = bool(msg.get("acted_upon"))

    # Root-cause event description. Prefer the selector's own
    # `root_cause_reason` verbatim — it carries the fallback marker
    # "(no non-aggregator ancestor; selected from full ancestor set)"
    # when relevant, so the user sees *why* a degenerate run picked an
    # aggregator. Only synthesize from event_id/agent when the reason
    # is missing (pre-selector records).
    root_summary = (report.get("root_cause_reason") or "").strip()
    if not root_summary:
        root_id = report.get("root_cause_event_id") or ""
        if root_id:
            root_ev = ev_map.get(root_id) or {}
            root_action = root_ev.get("action") or root_ev.get("event_type") or "event"
            root_agent = root_ev.get("agent_id") or "unknown"
            root_summary = f"{root_action} from {root_agent}"

    sentences: List[str] = []
    if top_agent:
        sentences.append(
            f"The outcome was mainly driven by <b>{top_agent}</b> "
            f"(responsibility {top_score:.2f})."
        )
    if tool_name and tool_score > 0:
        caller_str = f" called by {tool_caller}" if tool_caller else ""
        sentences.append(
            f"The most impactful tool was <code>{tool_name}</code>"
            f"{caller_str} (impact {tool_score:.2f})."
        )
    if msg_pair:
        msg_phrase = "was acted upon" if msg_acted else "carried the highest weight (heuristic only — not flagged acted_upon)"
        sentences.append(f"The most influential message was {msg_pair} and {msg_phrase}.")
    if root_summary:
        sentences.append(f"The root-cause event was <i>{root_summary}</i>.")

    # "Specialist A contributed little" — only when its responsibility is
    # below 0.20 AND it has no observable signals (no impactful tool, no
    # acted-upon message, no cited memory).
    a_score = float(scores.get("specialist_a", 0.0))
    if a_score < 0.20:
        from agentxai.ui.faithfulness_checks import _agent_observable_signals
        a_signals = _agent_observable_signals("specialist_a", xai, report)
        if not any(a_signals.values()):
            sentences.append(
                "Specialist A contributed little in this case — its symptom "
                "lookup produced no candidate conditions, no acted-upon "
                "message, and no cited memory."
            )

    if not sentences:
        return "The accountability report is empty for this task."
    return " ".join(sentences)


def _build_faithfulness_snapshot(
    record: Dict[str, Any],
) -> List[Tuple[str, str, str]]:
    """
    Produce a 5-row checklist of high-level faithfulness signals for the
    Summary card. Returns ``[(label, status, note)]`` where status is
    ``"pass" | "warn" | "fail" | "skip"``.

    Reuses the canonical checks in `agentxai.ui.faithfulness_checks` so the
    Summary card and the Accountability tab's panel stay consistent.
    """
    from agentxai.ui.faithfulness_checks import (
        check_impactful_tool_on_chain,
        check_no_undeserved_responsibility,
        check_rationale_cites_evidence,
        check_root_cause_not_aggregator,
        check_top_agent_has_signal,
    )

    xai = record.get("xai_data") or {}
    report = xai.get("accountability_report") or {}
    trajectory = xai.get("trajectory") or []

    rows: List[Tuple[str, str, str]] = [
        (
            "Top agent has observable signal",
            check_top_agent_has_signal(record).get("status", "skip"),
            "",
        ),
        (
            "Most-impactful tool is on causal path",
            check_impactful_tool_on_chain(record).get("status", "skip"),
            "",
        ),
        (
            "Root cause is not an aggregator",
            check_root_cause_not_aggregator(record).get("status", "skip"),
            "",
        ),
        (
            "Rationale references retrieved evidence",
            check_rationale_cites_evidence(record).get("status", "skip"),
            "",
        ),
        (
            "No high-responsibility agent with empty signals",
            check_no_undeserved_responsibility(record).get("status", "skip"),
            "",
        ),
    ]

    # Extra inline warning for the user-mentioned bug pattern: root cause
    # ended up on a synthesizer aggregation event despite the selector
    # filter. Reads the trajectory to spot the specific aggregator names.
    root_id = report.get("root_cause_event_id") or ""
    if root_id:
        ev = next((e for e in trajectory if e.get("event_id") == root_id), None)
        if ev and _is_aggregator_event(ev):
            rows.append((
                "⚠ Root cause is an aggregator event",
                "warn",
                f"action='{ev.get('action', '?')}' — selector fell back",
            ))
    return rows


def _render_summary_card(
    eyebrow: str, body_html: str, variant: str = "accent",
) -> None:
    """Render one of the five Summary-page narrative cards."""
    st.markdown(
        f'<div class="xai-card is-{variant}" style="margin:14px 0;">'
        f'<div class="xai-card-eyebrow">{eyebrow}</div>'
        f'<div class="xai-card-sub" style="font-size:0.95rem;line-height:1.5;'
        f'color:#222;">{body_html}</div>'
        '</div>',
        unsafe_allow_html=True,
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

    # Question-type pill (set by the heuristic classifier in
    # `agentxai.data.question_classifier` at task-creation time). Older
    # records without the field render as "unknown".
    question_type = (input_data.get("question_type") or "unknown")
    qtype_pill = (
        f'<span class="xai-pill" title="Heuristic question-type label '
        f'used to weight agent responsibility.">'
        f'Type: <b>&nbsp;{question_type}</b></span>'
    )

    # --- Hero --------------------------------------------------------------
    st.markdown(
        f"""
        <div class="xai-hero">
          <div class="xai-hero-eyebrow">Case overview · task {task_id[:12]}</div>
          <h1 class="xai-hero-title">{final_diagnosis}</h1>
          <div class="xai-hero-meta">
            {correct_pill}
            {qtype_pill}
            <span class="xai-pill">Ground truth: <b>&nbsp;{correct_answer}</b></span>
            <span class="xai-pill">Confidence: <b>&nbsp;{conf_str}</b></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Aggregate counts pulled once for the rest of the page.
    trajectory = xai.get("trajectory", []) or []
    plans = xai.get("plans", []) or []
    tool_calls = xai.get("tool_calls", []) or []
    memory_diffs = xai.get("memory_diffs", []) or []
    messages = xai.get("messages", []) or []

    # ─── Card 1: Final Answer ───────────────────────────────────────────────
    _render_summary_card(
        "Final answer",
        _build_final_answer_sentence(record),
        variant=("success" if correct else "error" if correct is False else "accent"),
    )

    # ─── Card 2: Pipeline Story ─────────────────────────────────────────────
    bullets = _build_pipeline_story_bullets(record)
    bullets_html = "".join(f"<li style='margin:4px 0;'>{b}</li>" for b in bullets)
    _render_summary_card(
        "Pipeline story",
        f"<ul style='padding-left:20px;margin:4px 0;'>{bullets_html}</ul>",
    )

    # ─── Card 3: What actually mattered ─────────────────────────────────────
    # Banner first so users see the "re-run needed" warning before they
    # try to act on the (possibly stale) attribution prose below it.
    _render_staleness_banner(report)
    _render_summary_card(
        "What actually mattered",
        _build_what_mattered_paragraph(record),
    )

    # ─── Card 4: Evidence used ──────────────────────────────────────────────
    evidence_items = _get_supporting_evidence(record, top_n=3)
    if evidence_items:
        rows = []
        for ev in evidence_items:
            doc_id = str(ev.get("doc_id") or "?")
            source = str(ev.get("source_file") or "?")
            try:
                score = float(ev.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            snippet = str(ev.get("snippet") or ev.get("text") or "").strip()
            if len(snippet) > 220:
                snippet = snippet[:217] + "…"
            badge = (
                "<span style='color:#06A77D;'>✓ used in rationale</span>"
                if ev.get("used_in_rationale")
                else "<span style='color:#888;'>· retrieved but not cited</span>"
            )
            rows.append(
                "<div style='padding:8px 0;border-bottom:1px solid #eee;'>"
                f"<div style='font-size:0.9rem;'><code>{doc_id}</code> · "
                f"{source} · score {score:.2f} · {badge}</div>"
                + (f"<div style='color:#444;font-size:0.88rem;margin-top:4px;'>{snippet}</div>"
                   if snippet else "")
                + "</div>"
            )
        _render_summary_card("Evidence used", "".join(rows))
    else:
        _render_summary_card(
            "Evidence used",
            "<i>No retrieved evidence available for this task.</i>",
        )

    # ─── Card 5: Faithfulness snapshot ──────────────────────────────────────
    snapshot_rows = _build_faithfulness_snapshot(record)
    icons = {"pass": "✓", "warn": "⚠", "fail": "✗", "skip": "⊘"}
    colors = {
        "pass": "#06A77D", "warn": "#F4A261",
        "fail": "#E63946", "skip": "#888888",
    }
    snapshot_html = ""
    for label, status, note in snapshot_rows:
        icon = icons.get(status, "·")
        color = colors.get(status, "#888")
        note_html = (
            f" <span style='color:#888;font-size:0.85rem;'>— {note}</span>"
            if note else ""
        )
        snapshot_html += (
            "<div style='padding:4px 0;'>"
            f"<span style='color:{color};font-weight:bold;'>{icon}</span> "
            f"{label}{note_html}"
            "</div>"
        )
    _render_summary_card(
        "Faithfulness snapshot",
        snapshot_html
        + "<div style='margin-top:8px;font-size:0.78rem;color:#888;'>"
        "Heuristic checks. See the Accountability tab for the full panel."
        "</div>",
    )

    # ─── The case (patient text + answer options) ───────────────────────────
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
            predicted_letter = system_output.get("predicted_letter") or ""
            # Index option_analysis by letter so we can splice the
            # Synthesizer's per-option reason next to each option line.
            option_analysis_raw = system_output.get("option_analysis") or []
            analysis_by_letter: Dict[str, Dict[str, Any]] = {
                str(o.get("letter", "")).upper(): o
                for o in option_analysis_raw
                if isinstance(o, dict) and o.get("letter")
            }
            for k, v in options.items():
                if not v:
                    continue
                k_up = str(k).upper()
                is_truth = (k == correct_answer)
                is_pick = (k_up == str(predicted_letter).upper())
                marker = "🟢 " if is_truth else ""
                pick_tag = "  _← model picked_" if is_pick and not is_truth else ""
                truth_tag = "  _← ground truth_" if is_truth else ""
                st.markdown(f"- **{k}.** {marker}{v}{truth_tag}{pick_tag}")
                # Per-option Synthesizer rationale, when present. Verdict
                # colour-codes the line so the user can scan the table at a
                # glance: green=correct pick, red=incorrect dismissed,
                # yellow=partial.
                entry = analysis_by_letter.get(k_up)
                if entry and entry.get("reason"):
                    verdict = (entry.get("verdict") or "").lower()
                    icon = {
                        "correct": "✓",
                        "incorrect": "✗",
                        "partial": "~",
                    }.get(verdict, "·")
                    st.caption(f"&nbsp;&nbsp;&nbsp;{icon} {entry['reason']}")

            # Evidence cards: each retrieved Specialist-B doc with its
            # source file, FAISS score, snippet, and a badge if it appears
            # in the Synthesizer's rationale.
            _render_evidence_cards(
                top_evidence=_extract_top_evidence(xai),
                used_ids=system_output.get("supporting_evidence_ids") or [],
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

        # Heuristic confidence breakdown — only renders when the system
        # output carries the new `confidence_factors` key (older records
        # silently skip this panel).
        _render_confidence_factors_panel(
            confidence=confidence,
            factors=system_output.get("confidence_factors") or {},
        )

        gt_explanation = ground_truth.get("explanation") or ""
        if gt_explanation:
            with st.expander("Ground-truth explanation"):
                st.write(gt_explanation)

    # Responsibility distribution (kept as a small bar widget for users
    # who want the at-a-glance ranking; the prose lives in Card 3).
    scores: Dict[str, float] = report.get("agent_responsibility_scores") or {}
    if scores:
        with st.expander("Agent responsibility scores", expanded=False):
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
    # Kept inside an expander since the prose card above already names the
    # top agent / tool / message / root cause. Power users who want the
    # raw ids can still pop these open.
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
                    _tool_display_name(tool.get("tool_name")),
                    (
                        f"Called by {tool.get('called_by', '?')} · "
                        f"impact {impact:.2f}"
                    ),
                    variant="success",
                )
            else:
                tool_card = _info_card(
                    "Most-impactful tool call",
                    f"{_tool_display_name(tool.get('tool_name'))} · impact 0.00",
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

    with st.expander("Key attributions (raw ids)", expanded=False):
        c1, c2 = st.columns(2)
        c1.markdown(agent_card, unsafe_allow_html=True)
        c2.markdown(root_card, unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        c3.markdown(tool_card, unsafe_allow_html=True)
        c4.markdown(msg_card, unsafe_allow_html=True)

        if report.get("plan_deviation_summary"):
            st.markdown("**Plan deviation summary**")
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
    r = requests.get(
        _url(path), params=params or None,
        headers=_api_headers(), timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: Dict[str, Any]) -> Any:
    r = requests.post(
        _url(path), json=payload,
        headers=_api_headers(), timeout=HTTP_TIMEOUT * 10,
    )
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

    # Surface the display-alias rule so reviewers aren't misled by stored
    # tool names. We only show the note when an aliased tool actually
    # appeared in this task — keeps the UI quiet otherwise.
    aliased_present = sorted({
        c.get("tool_name", "")
        for c in calls
        if c.get("tool_name") in _TOOL_DISPLAY_OVERRIDES
    })
    if aliased_present:
        notes = " · ".join(
            f"`{name}` → **{_TOOL_DISPLAY_OVERRIDES[name]}**"
            for name in aliased_present
        )
        st.caption(
            f"ℹ Display alias: {notes}. The stored `tool_name` is preserved "
            "for backward compatibility with existing records."
        )

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
        f"#{i + 1} · {_tool_display_name(c.get('tool_name'))} · "
        f"called_by={c.get('called_by', '?')} · "
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


def _memory_usage_badge(usage: Optional[Dict[str, Any]]) -> str:
    """Inline badge shown next to each memory write key."""
    if not usage:
        return ""
    score = float(usage.get("influence_score") or 0.0)
    used = bool(usage.get("used_in_final_answer"))
    if used and score >= 0.5:
        return f"`✓ used (influence {score:.2f})`"
    if used:
        return f"`~ partially used (influence {score:.2f})`"
    return "`✗ not cited in rationale`"


def _render_memory_usage_panel(memory_usage: List[Dict[str, Any]]) -> None:
    """
    Show the per-(owner, key) usage table at the top of the Memory tab.

    Renders as a sortable dataframe with columns: agent, key, read_by,
    used_in_final_answer, influence_score. When the report carries no
    usage records (older runs before the attribution pass existed) the
    panel is silently skipped.
    """
    if not memory_usage:
        return
    rows = []
    for u in memory_usage:
        rows.append({
            "agent": u.get("agent_id", ""),
            "key":   u.get("key", ""),
            "read_by": ", ".join(u.get("read_by", []) or []),
            "used_in_final_answer": bool(u.get("used_in_final_answer")),
            "influence_score": float(u.get("influence_score") or 0.0),
        })
    if not rows:
        return
    st.markdown("**Memory usage attribution** — which writes the Synthesizer cited in its rationale")
    st.dataframe(
        rows,
        column_config={
            "influence_score": st.column_config.ProgressColumn(
                "influence_score",
                help="Fraction of value tokens cited in the Synthesizer's rationale.",
                min_value=0.0, max_value=1.0, format="%.2f",
            ),
            "used_in_final_answer": st.column_config.CheckboxColumn(
                "used_in_final_answer",
                help="True if any token from this value appears in the rationale.",
            ),
        },
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Heuristic: substring match of value tokens against the rationale. "
        "Empty / numeric-only values score 0.0 by construction."
    )


def render_memory_tab(
    memory_diffs: List[Dict[str, Any]],
    trajectory: List[Dict[str, Any]],
    memory_usage: Optional[List[Dict[str, Any]]] = None,
) -> None:
    render_tab_summary("Memory", _summarize_memory(memory_diffs))
    if not memory_diffs:
        st.info("No memory writes recorded for this task.")
        return

    writes = [m for m in memory_diffs if (m.get("operation") or "").lower() == "write"]
    if not writes:
        st.info("Memory activity recorded, but no writes.")
        return

    # Render the per-key usage panel (read_by / used_in_final_answer /
    # influence_score) above the raw write log, when attribution data is
    # available on the accountability report.
    _render_memory_usage_panel(memory_usage or [])

    by_agent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in writes:
        by_agent[m.get("agent_id") or "unknown"].append(m)

    # Index usage records by (agent_id, key) for inline lookup per write.
    usage_index: Dict[tuple, Dict[str, Any]] = {
        (u.get("agent_id"), u.get("key")): u
        for u in (memory_usage or [])
    }

    for agent_id, agent_writes in by_agent.items():
        agent_writes.sort(key=lambda m: m.get("timestamp", 0))
        with st.expander(f"Agent · `{agent_id}` · {len(agent_writes)} write(s)", expanded=False):
            for i, m in enumerate(agent_writes):
                diff_id = m.get("diff_id", f"d{i}")
                key = m.get("key", "?")
                usage = usage_index.get((agent_id, key))
                badge = _memory_usage_badge(usage)
                st.markdown(
                    f"**{i + 1}. `{key}`** {badge} · "
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
                    event = _find_event(trajectory, trig_id)
                    label_action = (
                        (event or {}).get("action")
                        or (event or {}).get("event_type")
                        or "event"
                    )
                    if st.toggle(
                        f"🔗 Linked to **{label_action}** · `{trig_id[:8]}`",
                        key=f"mem_trig_{diff_id}",
                    ):
                        if event is None:
                            st.caption(
                                "Triggering event id present but not found "
                                "in this task's trajectory."
                            )
                        else:
                            st.json(event)
                else:
                    st.caption(
                        "⚠ outside_traced_action — this write fired with no "
                        "trajectory event in scope. If the write came from "
                        "agent code, wrap it in `self.traced_action(...)` "
                        "so it's attributed to a specific action."
                    )

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


def _render_faithfulness_panel(record: Dict[str, Any]) -> None:
    """
    Render the "Faithfulness Checks" panel inside the Accountability tab.

    Each check (defined in `agentxai.ui.faithfulness_checks`) is a sanity
    assertion over the stored XAI data — does the report's headline agree
    with the underlying signals? Results are colour-coded: green check =
    pass, yellow warning = questionable, red cross = failed, gray =
    "Not enough data".
    """
    from agentxai.ui.faithfulness_checks import (
        compute_faithfulness_checks,
        summarize_check_results,
    )

    results = compute_faithfulness_checks(record)
    if not results:
        return
    counts = summarize_check_results(results)

    st.markdown("**Faithfulness Checks**")
    st.caption(
        f"✓ {counts['pass']} passed · "
        f"⚠ {counts['warn']} flagged · "
        f"✗ {counts['fail']} failed · "
        f"⊘ {counts['skip']} insufficient data"
    )

    # Inline rendering — one row per check. We use raw HTML so the colour
    # markers are tight against the explanation rather than wrapped in
    # heavy st.success / st.warning / st.error boxes (which would make
    # the panel dwarf the rest of the tab).
    style_for = {
        "pass": ("#06A77D", "✓"),
        "warn": ("#F4A261", "⚠"),
        "fail": ("#E63946", "✗"),
        "skip": ("#888888", "⊘"),
    }
    for r in results:
        color, icon = style_for.get(r.get("status", "skip"), ("#888888", "·"))
        st.markdown(
            f"<div style='padding:6px 0; border-bottom:1px solid #eee;'>"
            f"<span style='color:{color}; font-weight:bold; font-size:1.05em; "
            f"display:inline-block; width:1.4em;'>{icon}</span>"
            f"<b>{r.get('name', '?')}</b> — "
            f"<span style='color:#444;'>{r.get('explanation', '')}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_accountability_tab(
    report: Optional[Dict[str, Any]],
    trajectory: List[Dict[str, Any]],
    tool_calls: List[Dict[str, Any]],
    record: Optional[Dict[str, Any]] = None,
) -> None:
    render_tab_summary("Accountability", _summarize_accountability(report))
    if not report:
        st.info("No accountability report recorded for this task.")
        return

    # Surface stale-report warnings BEFORE the headline explanation so
    # users don't trust a fallback-marked root cause unconditionally.
    _render_staleness_banner(report)

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

    # Faithfulness Checks — sanity assertions over the report. Rendered last
    # so the existing layout above is unchanged for users with older records.
    if record is not None:
        st.divider()
        _render_faithfulness_panel(record)


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
        memory_usage = (accountability_report or {}).get("memory_usage", []) or []
        render_memory_tab(memory_diffs, trajectory, memory_usage)
    with tabs[4]:
        render_communication_tab(messages)
    with tabs[5]:
        render_causality_tab(causal_graph, trajectory, accountability_report)
    with tabs[6]:
        render_accountability_tab(
            accountability_report, trajectory, tool_calls, record=record,
        )


if __name__ == "__main__":
    main()
