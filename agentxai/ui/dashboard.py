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


# ---------------------------------------------------------------------------
# Sidebar — task selector + "Run new task"
# ---------------------------------------------------------------------------

def render_sidebar() -> Optional[str]:
    st.sidebar.title("AgentXAI")
    st.sidebar.caption(f"API: `{API_BASE}`")

    # Task list
    try:
        data = api_get("/tasks", per_page=200)
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
        st.rerun()

    st.sidebar.divider()

    # Run new task
    with st.sidebar.expander("▶ Run new task", expanded=not items):
        st.caption(
            "Paste a MedQA record as JSON. Shape: "
            "`{input: {...}, ground_truth: {...}}`."
        )
        default_record = json.dumps(
            {
                "input": {
                    "patient_case": "",
                    "answer_options": {"A": "", "B": "", "C": "", "D": "", "E": ""},
                },
                "ground_truth": {"correct_answer": "A", "explanation": ""},
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
            st.rerun()

    # Honor a newly-run task on next render.
    preferred = st.session_state.pop("preferred_task_id", None)
    if preferred and preferred in {it["task_id"] for it in items}:
        return preferred
    return selected_task_id


# ---------------------------------------------------------------------------
# Header strip
# ---------------------------------------------------------------------------

def render_header(record: Dict[str, Any]) -> None:
    task_id = record.get("task_id", "—")
    system_output = record.get("system_output") or {}
    ground_truth = record.get("ground_truth") or {}

    final_diagnosis = system_output.get("final_diagnosis") or "—"
    correct_answer = ground_truth.get("correct_answer") or "—"
    confidence = system_output.get("confidence")
    correct = system_output.get("correct")
    badge = "✅" if correct else ("❌" if correct is False else "·")

    conf_str = f"{float(confidence):.2f}" if isinstance(confidence, (int, float)) else "—"

    c1, c2, c3, c4, c5 = st.columns([2, 3, 2, 1, 1])
    c1.metric("task_id", task_id[:8])
    c2.metric("Final diagnosis", final_diagnosis)
    c3.metric("Ground truth", correct_answer)
    c4.metric("Result", badge)
    c5.metric("Confidence", conf_str)


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
        cf_record = api_get(f"/tasks/{cf_id}")
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

    task_id = render_sidebar()
    if not task_id:
        st.title("AgentXAI")
        st.info("Select a task from the sidebar, or run a new one.")
        return

    try:
        record = api_get(f"/tasks/{task_id}")
    except Exception as e:
        st.error(f"Could not fetch /tasks/{task_id}: {e}")
        return

    render_header(record)
    st.divider()

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
