"""Deep diagnostic: dump the full XAI state of the most recent task."""
from dotenv import load_dotenv

load_dotenv(override=True)

import json
from sqlalchemy import text

from agentxai.store.trajectory_store import TrajectoryStore


def _pp(obj, indent=2):
    return json.dumps(obj, indent=indent, default=str, ensure_ascii=False)


def main() -> None:
    store = TrajectoryStore()

    with store._engine.connect() as conn:
        tasks = conn.execute(text(
            "SELECT task_id, created_at FROM tasks ORDER BY created_at DESC LIMIT 5"
        )).fetchall()

    if not tasks:
        print("No tasks found.")
        return

    print("=" * 80)
    print(f"RECENT TASKS ({len(tasks)})")
    print("=" * 80)
    for tid, ts in tasks:
        print(f"  {ts}  {tid}")

    tid = tasks[0][0]
    print(f"\nInspecting most recent task: {tid}\n")

    rec = store.get_full_record(tid)
    xai = rec.xai_data

    print("=" * 80)
    print("SYSTEM OUTPUT")
    print("=" * 80)
    for k, v in rec.system_output.items():
        print(f"  {k}: {v!r}")

    print("\n" + "=" * 80)
    print(f"TRAJECTORY  ({len(xai.trajectory)} events)")
    print("=" * 80)
    for i, e in enumerate(xai.trajectory):
        print(f"  {i+1:2}. [{e.agent_id:12}] {e.event_type:8} {e.action:30} -> {e.outcome!r}")

    print("\n" + "=" * 80)
    print(f"PLANS  ({len(xai.plans)} total)")
    print("=" * 80)
    for p in xai.plans:
        print(f"  agent={p.agent_id}  plan_id={p.plan_id}")
        print(f"    intended: {p.intended_actions}")
        print(f"    actual:   {p.actual_actions}")
        print(f"    deviations:        {p.deviations}")
        print(f"    deviation_reasons: {p.deviation_reasons}")
        print()

    print("=" * 80)
    print(f"TOOL CALLS  ({len(xai.tool_calls)} total)")
    print("=" * 80)
    for tc in xai.tool_calls:
        print(f"  [{tc.called_by}] {tc.tool_name}  id={tc.tool_call_id[:8]}")
        print(f"    inputs:  {_pp(tc.inputs)[:200]}")
        print(f"    outputs: {str(tc.outputs)[:200]}")
        print(f"    downstream_impact_score: {tc.downstream_impact_score}")
        print()

    print("=" * 80)
    print(f"MEMORY DIFFS  ({len(xai.memory_diffs)} total)")
    print("=" * 80)
    by_agent = {}
    for d in xai.memory_diffs:
        by_agent.setdefault(d.agent_id, []).append(d)
    for agent, diffs in by_agent.items():
        print(f"  {agent}:  {len(diffs)} diffs, keys={sorted({d.key for d in diffs})}")

    print("\n" + "=" * 80)
    print(f"MESSAGES  ({len(xai.messages)} total)")
    print("=" * 80)
    for m in xai.messages:
        print(f"  {m.sender:12} -> {m.receiver:12}  type={m.message_type:12}  acted_upon={m.acted_upon}")

    print("\n" + "=" * 80)
    print(f"CAUSAL GRAPH  nodes={len(xai.causal_graph.nodes)}  edges={len(xai.causal_graph.edges)}")
    print("=" * 80)
    for e in xai.causal_graph.edges[:10]:
        print(f"  {e.cause_event_id[:8]} -> {e.effect_event_id[:8]}   strength={e.causal_strength:.2f}  type={e.causal_type}")
    if len(xai.causal_graph.edges) > 10:
        print(f"  ... and {len(xai.causal_graph.edges) - 10} more")

    print("\n" + "=" * 80)
    print("ACCOUNTABILITY REPORT")
    print("=" * 80)
    ar = xai.accountability_report
    if ar is None:
        print("  (none)")
    else:
        print(f"  responsibility_scores: {ar.agent_responsibility_scores}")
        print(f"  root_cause_event_id:   {ar.root_cause_event_id}")
        print(f"  causal_chain:          {ar.causal_chain}")
        print(f"  most_impactful_tool:   {ar.most_impactful_tool_call_id}")
        print(f"  most_influential_msg:  {ar.most_influential_message_id}")
        print(f"  plan_deviation_summary:")
        for line in (ar.plan_deviation_summary or []):
            print(f"    - {line}")
        print(f"  one_line_explanation:  {ar.one_line_explanation!r}")


if __name__ == "__main__":
    main()
