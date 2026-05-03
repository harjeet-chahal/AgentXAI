import json
import sqlite3
from pathlib import Path
from dataclasses import asdict, is_dataclass

from agentxai.store.trajectory_store import TrajectoryStore

DB_PATH = Path("agentxai/data/agentxai.db").resolve()
print("Using DB:", DB_PATH)

store = TrajectoryStore(db_url=f"sqlite:///{DB_PATH}")

def to_jsonable(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("SELECT task_id FROM tasks")
task_ids = [row[0] for row in cur.fetchall()]
conn.close()

print(f"Found {len(task_ids)} tasks")

records = []

for task_id in task_ids:
    try:
        record = store.get_full_record(task_id)
        records.append(to_jsonable(record))
        print("Exported:", task_id)
    except Exception as e:
        print(f"Skipping {task_id}: {e}")

with open("exported_agentxai_records.json", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False, default=str)

print(f"Saved {len(records)} records to exported_agentxai_records.json")