"""Quick diagnostic: dump the most recent task's synthesizer output."""
from dotenv import load_dotenv

load_dotenv(override=True)

from sqlalchemy import text

from agentxai.store.trajectory_store import TrajectoryStore


def main() -> None:
    store = TrajectoryStore()
    with store._engine.connect() as conn:
        tid = conn.execute(text(
            "SELECT task_id FROM tasks ORDER BY created_at DESC LIMIT 1"
        )).scalar()

    if not tid:
        print("No tasks found in the store.")
        return

    rec = store.get_full_record(tid)

    print(f"task_id: {tid}")
    print("\n=== system_output ===")
    for k, v in rec.system_output.items():
        print(f"  {k}: {v!r}")

    print("\n=== synthesizer final_output (from memory_diffs) ===")
    found = False
    for d in rec.xai_data.memory_diffs:
        if d.agent_id == "synthesizer" and d.key == "final_output":
            print(d.value_after)
            found = True
    if not found:
        print("(no synthesizer.final_output memory write)")

    print("\n=== last 10 trajectory events ===")
    for e in rec.xai_data.trajectory[-10:]:
        print(f"  [{e.agent_id}] {e.event_type} {e.action} -> {e.outcome!r}")


if __name__ == "__main__":
    main()
