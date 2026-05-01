"""
Pillar 8 — Counterfactual Engine.

Three perturbation types — tool output (Type 1), agent memory / output
(Type 2), and inter-agent message (Type 3). Each perturbation re-runs the
pipeline from the perturbation point via `Pipeline.resume_from(...)`,
compares the resulting diagnosis + confidence to the original run, and
writes a row to the `counterfactual_runs` SQLite table.

The pipeline itself is not implemented in this module; we only declare the
`Pipeline` Protocol that the Phase-5 runner must satisfy.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from sqlalchemy import text

from agentxai.data.schemas import ToolUseEvent
from agentxai.store.trajectory_store import TrajectoryStore


# ---------------------------------------------------------------------------
# Pipeline resume protocol (wired up in Phase 5)
# ---------------------------------------------------------------------------

@runtime_checkable
class Pipeline(Protocol):
    """
    Interface a pipeline runner must satisfy to support counterfactual re-runs.

    `state_snapshot` is an opaque dict describing the pre-perturbation state;
    `overrides` carries the perturbation to inject. The runner decides which
    downstream agents to re-execute based on the overrides. The returned dict
    must carry at least `final_diagnosis` (str) and `confidence` (float).
    """

    def resume_from(
        self,
        state_snapshot: Dict[str, Any],
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# counterfactual_runs table DDL
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS counterfactual_runs (
    run_id                 TEXT PRIMARY KEY,
    task_id                TEXT NOT NULL,
    perturbation_type      TEXT NOT NULL,
    target_id              TEXT NOT NULL,
    baseline_value_json    TEXT NOT NULL,
    original_outcome_json  TEXT NOT NULL,
    perturbed_outcome_json TEXT NOT NULL,
    outcome_delta          REAL NOT NULL
)
"""

_INSERT_SQL = (
    "INSERT INTO counterfactual_runs "
    "(run_id, task_id, perturbation_type, target_id, baseline_value_json, "
    " original_outcome_json, perturbed_outcome_json, outcome_delta) "
    "VALUES (:run_id, :task_id, :ptype, :tid, :bv, :oo, :po, :delta)"
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CounterfactualEngine:
    """Run Type-1/2/3 perturbations, score the outcome delta, log the run."""

    def __init__(
        self,
        store: TrajectoryStore,
        pipeline: Pipeline,
        task_id: str,
        state_snapshot: Optional[Dict[str, Any]] = None,
        original_output: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.store = store
        self.pipeline = pipeline
        self.task_id = task_id
        self.state_snapshot: Dict[str, Any] = dict(state_snapshot or {})
        self.original_output: Dict[str, Any] = dict(original_output or {})
        self._ensure_table()

    # ------------------------------------------------------------------
    # Public perturbation API
    # ------------------------------------------------------------------

    def perturb_tool_output(self, tool_call_id: str) -> float:
        """Type 1 — neutralize a tool's output; return downstream_impact_score."""
        tool = self._find_tool_call(tool_call_id)
        baseline = _neutral_baseline(tool.outputs)
        perturbed = self._resume({"tool_output": {tool_call_id: baseline}})
        delta = _outcome_delta(self.original_output, perturbed)
        self._log_run(
            perturbation_type="tool_output",
            target_id=tool_call_id,
            baseline_value=baseline,
            perturbed_output=perturbed,
            delta=delta,
        )
        return delta

    def perturb_agent_output(self, agent_id: str) -> float:
        """Type 2 — neutralize a specialist's memory final state; return responsibility."""
        baseline: Dict[str, Any] = {}
        perturbed = self._resume({"agent_memory": {agent_id: baseline}})
        delta = _outcome_delta(self.original_output, perturbed)
        self._log_run(
            perturbation_type="agent_output",
            target_id=agent_id,
            baseline_value=baseline,
            perturbed_output=perturbed,
            delta=delta,
        )
        return delta

    def perturb_message(self, message_id: str) -> Tuple[bool, str]:
        """Type 3 — neutralize a message; return (changed_behavior, description)."""
        baseline = {"info": "no information"}
        perturbed = self._resume({"message_content": {message_id: baseline}})
        delta = _outcome_delta(self.original_output, perturbed)
        changed = delta > 0.0
        description = _describe_change(self.original_output, perturbed, changed)
        self._log_run(
            perturbation_type="message",
            target_id=message_id,
            baseline_value=baseline,
            perturbed_output=perturbed,
            delta=delta,
        )
        return changed, description

    # ------------------------------------------------------------------
    # Read helper (for tests / dashboards)
    # ------------------------------------------------------------------

    def list_runs(self) -> List[Dict[str, Any]]:
        with self.store._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT run_id, task_id, perturbation_type, target_id, "
                    "baseline_value_json, original_outcome_json, "
                    "perturbed_outcome_json, outcome_delta "
                    "FROM counterfactual_runs WHERE task_id = :tid"
                ),
                {"tid": self.task_id},
            ).fetchall()
        return [
            {
                "run_id": r[0],
                "task_id": r[1],
                "perturbation_type": r[2],
                "target_id": r[3],
                "baseline_value": json.loads(r[4]),
                "original_outcome": json.loads(r[5]),
                "perturbed_outcome": json.loads(r[6]),
                "outcome_delta": r[7],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resume(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        out = self.pipeline.resume_from(self.state_snapshot, overrides)
        return dict(out or {})

    def _find_tool_call(self, tool_call_id: str) -> ToolUseEvent:
        record = self.store.get_full_record(self.task_id)
        for tc in record.xai_data.tool_calls:
            if tc.tool_call_id == tool_call_id:
                return tc
        raise KeyError(
            f"Tool call {tool_call_id!r} not found for task {self.task_id!r}."
        )

    def _ensure_table(self) -> None:
        with self.store._engine.connect() as conn:
            conn.execute(text(_CREATE_TABLE_SQL))
            conn.commit()

    def _log_run(
        self,
        *,
        perturbation_type: str,
        target_id: str,
        baseline_value: Any,
        perturbed_output: Dict[str, Any],
        delta: float,
    ) -> str:
        run_id = str(uuid.uuid4())
        with self.store._engine.connect() as conn:
            conn.execute(
                text(_INSERT_SQL),
                {
                    "run_id": run_id,
                    "task_id": self.task_id,
                    "ptype": perturbation_type,
                    "tid": target_id,
                    "bv": json.dumps(baseline_value, ensure_ascii=False),
                    "oo": json.dumps(self.original_output, ensure_ascii=False),
                    "po": json.dumps(perturbed_output, ensure_ascii=False),
                    "delta": float(delta),
                },
            )
            conn.commit()
        return run_id


# ---------------------------------------------------------------------------
# Baseline + outcome-comparison helpers
# ---------------------------------------------------------------------------

def _neutral_baseline(value: Any) -> Any:
    """Neutral zero-information baseline matching the shape of `value`."""
    if isinstance(value, dict):
        return {}
    if isinstance(value, list):
        return []
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return 0.0
    return {}


_DX_WEIGHT = 0.6
_CONF_WEIGHT = 0.4


def _outcome_delta(orig: Dict[str, Any], perturbed: Dict[str, Any]) -> float:
    """
    Graded counterfactual outcome delta in [0, 1].

    Combines two signals as a weighted sum so neither saturates the score on
    its own — a diagnosis flip with unchanged confidence no longer maxes out
    at 1.0, leaving room for confidence shifts to differentiate.

      score = 0.6 * dx_changed (binary)  +  0.4 * conf_delta (continuous)

    Reference points:
        identical                      → 0.00
        dx unchanged, conf delta 0.50  → 0.20
        dx changed,   conf unchanged   → 0.60
        dx changed,   conf delta 0.50  → 0.80
        dx changed,   conf delta 1.00  → 1.00
    """
    orig_dx = orig.get("final_diagnosis")
    new_dx = perturbed.get("final_diagnosis")
    dx_changed = 0.0 if orig_dx == new_dx else 1.0
    try:
        conf_delta = abs(
            float(orig.get("confidence", 0.0))
            - float(perturbed.get("confidence", 0.0))
        )
    except (TypeError, ValueError):
        conf_delta = 0.0
    conf_delta = max(0.0, min(1.0, conf_delta))
    return min(1.0, _DX_WEIGHT * dx_changed + _CONF_WEIGHT * conf_delta)


def _describe_change(
    orig: Dict[str, Any],
    perturbed: Dict[str, Any],
    changed: bool,
) -> str:
    if not changed:
        return "no behavior change"
    return (
        f"diagnosis {orig.get('final_diagnosis')!r} → "
        f"{perturbed.get('final_diagnosis')!r} "
        f"(confidence {orig.get('confidence', 0.0)} → "
        f"{perturbed.get('confidence', 0.0)})"
    )
