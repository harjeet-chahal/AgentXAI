"""
SQLite-backed persistence layer for all AgentXAI schemas.

One table per schema, all sharing task_id as a foreign key back to the tasks
table.  JSON blobs are used for any dict/list field so the table DDL stays
stable as the dataclass evolves.

Default database: agentxai/data/indices/../agentxai.db
  → resolves to agentxai/data/agentxai.db

Usage
-----
    store = TrajectoryStore()          # opens / creates DB
    store.save_task(record)
    store.save_event(task_id, event)
    record = store.get_full_record(task_id)
"""

from __future__ import annotations

import json
import pathlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from agentxai.data.schemas import (
    AccountabilityReport,
    AgentMessage,
    AgentPlan,
    AgentXAIRecord,
    CausalEdge,
    CausalGraph,
    MemoryDiff,
    ToolUseEvent,
    TrajectoryEvent,
    XAIData,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT    = pathlib.Path(__file__).resolve().parents[2]
_DB_PATH = _ROOT / "agentxai" / "data" / "agentxai.db"


# ---------------------------------------------------------------------------
# ORM base + helpers
# ---------------------------------------------------------------------------

class _Base(DeclarativeBase):
    pass


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _loads(s: Optional[str]) -> Any:
    if s is None:
        return None
    return json.loads(s)


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# ORM table definitions
# ---------------------------------------------------------------------------

class _Task(_Base):
    __tablename__ = "tasks"

    task_id          = Column(String, primary_key=True)
    source           = Column(String, nullable=False, default="medqa")
    input_json       = Column(Text,   nullable=False, default="{}")
    ground_truth_json= Column(Text,   nullable=False, default="{}")
    system_output_json=Column(Text,   nullable=False, default="{}")
    created_at       = Column(DateTime(timezone=True), nullable=False,
                              default=_now_dt)


class _TrajectoryEvent(_Base):
    __tablename__ = "trajectory_events"

    event_id          = Column(String, primary_key=True)
    task_id           = Column(String, ForeignKey("tasks.task_id"), nullable=False, index=True)
    timestamp         = Column(Float,  nullable=False)
    agent_id          = Column(String, nullable=False, default="")
    event_type        = Column(String, nullable=False, default="")
    state_before_json = Column(Text,   nullable=False, default="{}")
    action            = Column(String, nullable=False, default="")
    action_inputs_json= Column(Text,   nullable=False, default="{}")
    state_after_json  = Column(Text,   nullable=False, default="{}")
    outcome           = Column(String, nullable=False, default="")


class _AgentPlan(_Base):
    __tablename__ = "agent_plans"

    plan_id                 = Column(String, primary_key=True)
    task_id                 = Column(String, ForeignKey("tasks.task_id"), nullable=False, index=True)
    agent_id                = Column(String, nullable=False, default="")
    timestamp               = Column(Float,  nullable=False)
    intended_actions_json   = Column(Text,   nullable=False, default="[]")
    actual_actions_json     = Column(Text,   nullable=False, default="[]")
    deviations_json         = Column(Text,   nullable=False, default="[]")
    deviation_reasons_json  = Column(Text,   nullable=False, default="[]")


class _ToolUseEvent(_Base):
    __tablename__ = "tool_use_events"

    tool_call_id              = Column(String, primary_key=True)
    task_id                   = Column(String, ForeignKey("tasks.task_id"), nullable=False, index=True)
    tool_name                 = Column(String, nullable=False, default="")
    called_by                 = Column(String, nullable=False, default="")
    timestamp                 = Column(Float,  nullable=False)
    inputs_json               = Column(Text,   nullable=False, default="{}")
    outputs_json              = Column(Text,   nullable=False, default="{}")
    duration_ms               = Column(Float,  nullable=False, default=0.0)
    downstream_impact_score   = Column(Float,  nullable=False, default=0.0)
    counterfactual_run_id     = Column(String, nullable=False, default="")


class _MemoryDiff(_Base):
    __tablename__ = "memory_diffs"

    diff_id                 = Column(String, primary_key=True)
    task_id                 = Column(String, ForeignKey("tasks.task_id"), nullable=False, index=True)
    agent_id                = Column(String, nullable=False, default="")
    timestamp               = Column(Float,  nullable=False)
    operation               = Column(String, nullable=False, default="")
    key                     = Column(String, nullable=False, default="")
    value_before_json       = Column(Text,   nullable=True)
    value_after_json        = Column(Text,   nullable=True)
    triggered_by_event_id   = Column(String, nullable=False, default="")


class _AgentMessage(_Base):
    __tablename__ = "agent_messages"

    message_id                  = Column(String, primary_key=True)
    task_id                     = Column(String, ForeignKey("tasks.task_id"), nullable=False, index=True)
    sender                      = Column(String, nullable=False, default="")
    receiver                    = Column(String, nullable=False, default="")
    timestamp                   = Column(Float,  nullable=False)
    message_type                = Column(String, nullable=False, default="")
    content_json                = Column(Text,   nullable=False, default="{}")
    acted_upon                  = Column(Boolean,nullable=False, default=False)
    behavior_change_description = Column(Text,   nullable=False, default="")


class _CausalEdge(_Base):
    __tablename__ = "causal_edges"

    edge_id          = Column(String, primary_key=True)
    task_id          = Column(String, ForeignKey("tasks.task_id"), nullable=False, index=True)
    cause_event_id   = Column(String, nullable=False, default="")
    effect_event_id  = Column(String, nullable=False, default="")
    causal_strength  = Column(Float,  nullable=False, default=0.0)
    causal_type      = Column(String, nullable=False, default="")


class _AccountabilityReport(_Base):
    __tablename__ = "accountability_reports"

    task_id                       = Column(String, ForeignKey("tasks.task_id"),
                                           primary_key=True)
    final_outcome                 = Column(String, nullable=False, default="")
    outcome_correct               = Column(Boolean,nullable=False, default=False)
    agent_responsibility_scores_json = Column(Text, nullable=False, default="{}")
    root_cause_event_id           = Column(String, nullable=False, default="")
    causal_chain_json             = Column(Text,   nullable=False, default="[]")
    most_impactful_tool_call_id   = Column(String, nullable=False, default="")
    critical_memory_diffs_json    = Column(Text,   nullable=False, default="[]")
    most_influential_message_id   = Column(String, nullable=False, default="")
    plan_deviation_summary        = Column(Text,   nullable=False, default="")
    one_line_explanation          = Column(Text,   nullable=False, default="")


# ---------------------------------------------------------------------------
# Row → dataclass converters
# ---------------------------------------------------------------------------

def _row_to_event(row: _TrajectoryEvent) -> TrajectoryEvent:
    return TrajectoryEvent(
        event_id     = row.event_id,
        timestamp    = row.timestamp,
        agent_id     = row.agent_id,
        event_type   = row.event_type,
        state_before = _loads(row.state_before_json) or {},
        action       = row.action,
        action_inputs= _loads(row.action_inputs_json) or {},
        state_after  = _loads(row.state_after_json) or {},
        outcome      = row.outcome,
    )


def _row_to_plan(row: _AgentPlan) -> AgentPlan:
    return AgentPlan(
        plan_id          = row.plan_id,
        agent_id         = row.agent_id,
        timestamp        = row.timestamp,
        intended_actions = _loads(row.intended_actions_json) or [],
        actual_actions   = _loads(row.actual_actions_json) or [],
        deviations       = _loads(row.deviations_json) or [],
        deviation_reasons= _loads(row.deviation_reasons_json) or [],
    )


def _row_to_tool_call(row: _ToolUseEvent) -> ToolUseEvent:
    return ToolUseEvent(
        tool_call_id            = row.tool_call_id,
        tool_name               = row.tool_name,
        called_by               = row.called_by,
        timestamp               = row.timestamp,
        inputs                  = _loads(row.inputs_json) or {},
        outputs                 = _loads(row.outputs_json) or {},
        duration_ms             = row.duration_ms,
        downstream_impact_score = row.downstream_impact_score,
        counterfactual_run_id   = row.counterfactual_run_id,
    )


def _row_to_memory_diff(row: _MemoryDiff) -> MemoryDiff:
    return MemoryDiff(
        diff_id               = row.diff_id,
        agent_id              = row.agent_id,
        timestamp             = row.timestamp,
        operation             = row.operation,
        key                   = row.key,
        value_before          = _loads(row.value_before_json),
        value_after           = _loads(row.value_after_json),
        triggered_by_event_id = row.triggered_by_event_id,
    )


def _row_to_message(row: _AgentMessage) -> AgentMessage:
    return AgentMessage(
        message_id                  = row.message_id,
        sender                      = row.sender,
        receiver                    = row.receiver,
        timestamp                   = row.timestamp,
        message_type                = row.message_type,
        content                     = _loads(row.content_json) or {},
        acted_upon                  = bool(row.acted_upon),
        behavior_change_description = row.behavior_change_description,
    )


def _row_to_causal_edge(row: _CausalEdge) -> CausalEdge:
    return CausalEdge(
        edge_id         = row.edge_id,
        cause_event_id  = row.cause_event_id,
        effect_event_id = row.effect_event_id,
        causal_strength = row.causal_strength,
        causal_type     = row.causal_type,
    )


def _row_to_accountability(row: _AccountabilityReport) -> AccountabilityReport:
    return AccountabilityReport(
        task_id                     = row.task_id,
        final_outcome               = row.final_outcome,
        outcome_correct             = bool(row.outcome_correct),
        agent_responsibility_scores = _loads(row.agent_responsibility_scores_json) or {},
        root_cause_event_id         = row.root_cause_event_id,
        causal_chain                = _loads(row.causal_chain_json) or [],
        most_impactful_tool_call_id = row.most_impactful_tool_call_id,
        critical_memory_diffs       = _loads(row.critical_memory_diffs_json) or [],
        most_influential_message_id = row.most_influential_message_id,
        plan_deviation_summary      = row.plan_deviation_summary,
        one_line_explanation        = row.one_line_explanation,
    )


# ---------------------------------------------------------------------------
# TrajectoryStore
# ---------------------------------------------------------------------------

class TrajectoryStore:
    """
    SQLAlchemy-backed store for all AgentXAI trajectory data.

    Parameters
    ----------
    db_url : SQLAlchemy connection URL.  Defaults to the project-level SQLite file.
             Pass "sqlite:///:memory:" in tests.
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        if db_url is None:
            _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            db_url = f"sqlite:///{_DB_PATH}"

        self._engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},   # safe for single-threaded use
            echo=False,
        )
        # Enable WAL mode for better concurrent read performance on SQLite
        with self._engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.commit()

        _Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def save_task(self, record: AgentXAIRecord) -> None:
        """Upsert the top-level task row from an AgentXAIRecord."""
        with self._Session() as session:
            row = session.get(_Task, record.task_id)
            if row is None:
                row = _Task(task_id=record.task_id)
                session.add(row)
            row.source            = record.source
            row.input_json        = _dumps(record.input)
            row.ground_truth_json = _dumps(record.ground_truth)
            row.system_output_json= _dumps(record.system_output)
            session.commit()

    def save_event(self, task_id: str, event: TrajectoryEvent) -> None:
        """Insert or replace a TrajectoryEvent row."""
        with self._Session() as session:
            row = session.get(_TrajectoryEvent, event.event_id)
            if row is None:
                row = _TrajectoryEvent(event_id=event.event_id)
                session.add(row)
            row.task_id            = task_id
            row.timestamp          = event.timestamp
            row.agent_id           = event.agent_id
            row.event_type         = event.event_type
            row.state_before_json  = _dumps(event.state_before)
            row.action             = event.action
            row.action_inputs_json = _dumps(event.action_inputs)
            row.state_after_json   = _dumps(event.state_after)
            row.outcome            = event.outcome
            session.commit()

    def save_plan(self, task_id: str, plan: AgentPlan) -> None:
        """Insert or replace an AgentPlan row."""
        with self._Session() as session:
            row = session.get(_AgentPlan, plan.plan_id)
            if row is None:
                row = _AgentPlan(plan_id=plan.plan_id)
                session.add(row)
            row.task_id                = task_id
            row.agent_id               = plan.agent_id
            row.timestamp              = plan.timestamp
            row.intended_actions_json  = _dumps(plan.intended_actions)
            row.actual_actions_json    = _dumps(plan.actual_actions)
            row.deviations_json        = _dumps(plan.deviations)
            row.deviation_reasons_json = _dumps(plan.deviation_reasons)
            session.commit()

    def save_tool_call(self, task_id: str, tool_call: ToolUseEvent) -> None:
        """Insert or replace a ToolUseEvent row."""
        with self._Session() as session:
            row = session.get(_ToolUseEvent, tool_call.tool_call_id)
            if row is None:
                row = _ToolUseEvent(tool_call_id=tool_call.tool_call_id)
                session.add(row)
            row.task_id                 = task_id
            row.tool_name               = tool_call.tool_name
            row.called_by               = tool_call.called_by
            row.timestamp               = tool_call.timestamp
            row.inputs_json             = _dumps(tool_call.inputs)
            row.outputs_json            = _dumps(tool_call.outputs)
            row.duration_ms             = tool_call.duration_ms
            row.downstream_impact_score = tool_call.downstream_impact_score
            row.counterfactual_run_id   = tool_call.counterfactual_run_id
            session.commit()

    def save_memory_diff(self, task_id: str, diff: MemoryDiff) -> None:
        """Insert or replace a MemoryDiff row."""
        with self._Session() as session:
            row = session.get(_MemoryDiff, diff.diff_id)
            if row is None:
                row = _MemoryDiff(diff_id=diff.diff_id)
                session.add(row)
            row.task_id               = task_id
            row.agent_id              = diff.agent_id
            row.timestamp             = diff.timestamp
            row.operation             = diff.operation
            row.key                   = diff.key
            row.value_before_json     = _dumps(diff.value_before)
            row.value_after_json      = _dumps(diff.value_after)
            row.triggered_by_event_id = diff.triggered_by_event_id
            session.commit()

    def save_message(self, task_id: str, message: AgentMessage) -> None:
        """Insert or replace an AgentMessage row."""
        with self._Session() as session:
            row = session.get(_AgentMessage, message.message_id)
            if row is None:
                row = _AgentMessage(message_id=message.message_id)
                session.add(row)
            row.task_id                     = task_id
            row.sender                      = message.sender
            row.receiver                    = message.receiver
            row.timestamp                   = message.timestamp
            row.message_type                = message.message_type
            row.content_json                = _dumps(message.content)
            row.acted_upon                  = message.acted_upon
            row.behavior_change_description = message.behavior_change_description
            session.commit()

    def save_causal_edge(self, task_id: str, edge: CausalEdge) -> None:
        """Insert or replace a CausalEdge row."""
        with self._Session() as session:
            row = session.get(_CausalEdge, edge.edge_id)
            if row is None:
                row = _CausalEdge(edge_id=edge.edge_id)
                session.add(row)
            row.task_id         = task_id
            row.cause_event_id  = edge.cause_event_id
            row.effect_event_id = edge.effect_event_id
            row.causal_strength = edge.causal_strength
            row.causal_type     = edge.causal_type
            session.commit()

    def save_accountability_report(self, report: AccountabilityReport) -> None:
        """Insert or replace the AccountabilityReport for a task (keyed by task_id)."""
        with self._Session() as session:
            row = session.get(_AccountabilityReport, report.task_id)
            if row is None:
                row = _AccountabilityReport(task_id=report.task_id)
                session.add(row)
            row.final_outcome                    = report.final_outcome
            row.outcome_correct                  = report.outcome_correct
            row.agent_responsibility_scores_json = _dumps(report.agent_responsibility_scores)
            row.root_cause_event_id              = report.root_cause_event_id
            row.causal_chain_json                = _dumps(report.causal_chain)
            row.most_impactful_tool_call_id      = report.most_impactful_tool_call_id
            row.critical_memory_diffs_json       = _dumps(report.critical_memory_diffs)
            row.most_influential_message_id      = report.most_influential_message_id
            row.plan_deviation_summary           = report.plan_deviation_summary
            row.one_line_explanation             = report.one_line_explanation
            session.commit()

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    def get_full_record(self, task_id: str) -> AgentXAIRecord:
        """
        Reconstruct a complete AgentXAIRecord from the database.

        Raises
        ------
        KeyError if no task with that task_id exists.
        """
        with self._Session() as session:
            task_row = session.get(_Task, task_id)
            if task_row is None:
                raise KeyError(f"Task {task_id!r} not found in the store.")

            events = (
                session.query(_TrajectoryEvent)
                .filter_by(task_id=task_id)
                .order_by(_TrajectoryEvent.timestamp)
                .all()
            )
            plans = (
                session.query(_AgentPlan)
                .filter_by(task_id=task_id)
                .order_by(_AgentPlan.timestamp)
                .all()
            )
            tool_calls = (
                session.query(_ToolUseEvent)
                .filter_by(task_id=task_id)
                .order_by(_ToolUseEvent.timestamp)
                .all()
            )
            diffs = (
                session.query(_MemoryDiff)
                .filter_by(task_id=task_id)
                .order_by(_MemoryDiff.timestamp)
                .all()
            )
            messages = (
                session.query(_AgentMessage)
                .filter_by(task_id=task_id)
                .order_by(_AgentMessage.timestamp)
                .all()
            )
            edges = (
                session.query(_CausalEdge)
                .filter_by(task_id=task_id)
                .all()
            )
            ar_row = session.get(_AccountabilityReport, task_id)

        # Build the causal graph: nodes = deduplicated event_ids referenced by edges
        node_ids = []
        seen: set = set()
        for e in edges:
            for eid in (e.cause_event_id, e.effect_event_id):
                if eid and eid not in seen:
                    node_ids.append(eid)
                    seen.add(eid)

        causal_graph = CausalGraph(
            nodes=[_row_to_event(ev).event_id for ev in events] if not node_ids else node_ids,
            edges=[_row_to_causal_edge(e) for e in edges],
        )

        xai_data = XAIData(
            trajectory          = [_row_to_event(r) for r in events],
            plans               = [_row_to_plan(r) for r in plans],
            tool_calls          = [_row_to_tool_call(r) for r in tool_calls],
            memory_diffs        = [_row_to_memory_diff(r) for r in diffs],
            messages            = [_row_to_message(r) for r in messages],
            causal_graph        = causal_graph,
            accountability_report = (
                _row_to_accountability(ar_row) if ar_row is not None else None
            ),
        )

        return AgentXAIRecord(
            task_id      = task_row.task_id,
            source       = task_row.source,
            input        = _loads(task_row.input_json) or {},
            ground_truth = _loads(task_row.ground_truth_json) or {},
            system_output= _loads(task_row.system_output_json) or {},
            xai_data     = xai_data,
        )

    def list_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Return a summary list of the most-recently created tasks.

        Each entry: {"task_id", "source", "created_at", "final_outcome",
                     "outcome_correct"}.
        """
        with self._Session() as session:
            task_rows = (
                session.query(_Task)
                .order_by(_Task.created_at.desc())
                .limit(limit)
                .all()
            )
            result = []
            for t in task_rows:
                ar = session.get(_AccountabilityReport, t.task_id)
                result.append({
                    "task_id":        t.task_id,
                    "source":         t.source,
                    "created_at":     t.created_at.isoformat() if t.created_at else None,
                    "final_outcome":  ar.final_outcome if ar else None,
                    "outcome_correct":bool(ar.outcome_correct) if ar else None,
                })
        return result
