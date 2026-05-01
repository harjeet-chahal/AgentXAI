# regen_explanation.py (drop in repo root, run once)
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.accountability import AccountabilityReportGenerator

TASK_ID = "1e9b5b8c-06be-4247-b893-186ff2af9ffd"  # paste from sidebar

store = TrajectoryStore()
record = store.get_full_record(TASK_ID)
gen = AccountabilityReportGenerator(store=store, pipeline=None)
# pipeline=None → responsibility scores re-distributed equally; if you want
# them recomputed properly, pass your Pipeline instance.
gen.generate(
    task_id=TASK_ID,
    state_snapshot=None,
    original_output=dict(record.system_output or {}),
)
