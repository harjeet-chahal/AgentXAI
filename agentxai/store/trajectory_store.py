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
    Integer,
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
    MemoryUsage,
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
    root_cause_reason             = Column(Text,   nullable=False, default="")
    memory_usage_json             = Column(Text,   nullable=False, default="[]")
    evidence_used_by_final_answer_json = Column(Text, nullable=False, default="[]")
    most_supportive_evidence_ids_json  = Column(Text, nullable=False, default="[]")


class _ManualReview(_Base):
    """
    Manual XAI quality review (replaces the legacy raw-SQL `manual_reviews`).

    Two id columns reflecting the two natural keys at play:

      * ``medqa_task_id`` is the **stable MedQA record id** (e.g. "A00042")
        that the deterministic 500-record review split is keyed by. It's
        what reviewers actually rate — the question is the same regardless
        of how many times the pipeline has run for it. UNIQUE.
      * ``pipeline_task_id`` is a **soft FK to ``tasks.task_id``** (the
        per-run UUID generated by Pipeline). Nullable because reviews can
        be saved before the pipeline has been run for that record. When a
        run exists, the writer populates it so the review is reliably
        linked to the specific run that was inspected.

    The legacy `manual_reviews` table created by raw SQL inside the old
    Streamlit page is left in place untouched. ``migrate_legacy_manual_reviews``
    copies its rows into this table on first run; the legacy table stays
    readable so older eval scripts don't break mid-rollout.
    """

    __tablename__ = "manual_reviews_v2"

    review_id        = Column(Integer, primary_key=True, autoincrement=True)
    medqa_task_id    = Column(String, nullable=False, unique=True, index=True)
    pipeline_task_id = Column(
        String, ForeignKey("tasks.task_id"), nullable=True, index=True,
    )
    plausibility     = Column(Integer, nullable=True)
    completeness     = Column(Integer, nullable=True)
    specificity      = Column(Integer, nullable=True)
    causal_coherence = Column(Integer, nullable=True)
    notes            = Column(Text,    nullable=False, default="")
    status           = Column(String,  nullable=False, default="reviewed")
    reviewed_at      = Column(String,  nullable=False, default="")


# ---------------------------------------------------------------------------
# Row → dataclass converters
# ---------------------------------------------------------------------------

def _manual_review_to_dict(row: "_ManualReview") -> Dict[str, Any]:
    """Project a _ManualReview row to the public dict shape."""
    return {
        "review_id":        row.review_id,
        "medqa_task_id":    row.medqa_task_id,
        "pipeline_task_id": row.pipeline_task_id,
        "plausibility":     row.plausibility,
        "completeness":     row.completeness,
        "specificity":      row.specificity,
        "causal_coherence": row.causal_coherence,
        "notes":            row.notes or "",
        "status":           row.status,
        "reviewed_at":      row.reviewed_at,
        "source":           "v2",
    }


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
        root_cause_reason           = getattr(row, "root_cause_reason", "") or "",
        memory_usage                = [
            MemoryUsage.from_dict(u)
            for u in (_loads(getattr(row, "memory_usage_json", "") or "[]") or [])
        ],
        evidence_used_by_final_answer = list(
            _loads(getattr(row, "evidence_used_by_final_answer_json", "") or "[]") or []
        ),
        most_supportive_evidence_ids = list(
            _loads(getattr(row, "most_supportive_evidence_ids_json", "") or "[]") or []
        ),
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
        self._run_lightweight_migrations()
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)
        # One-shot copy of any pre-ORM `manual_reviews` rows into the
        # ORM-managed v2 table. Idempotent — see the method docstring.
        try:
            self.migrate_legacy_manual_reviews()
        except Exception:
            # Migration is best-effort; don't crash store init if a
            # legacy row is malformed. The v2 table still works for
            # new writes either way.
            pass

    def _run_lightweight_migrations(self) -> None:
        """
        Idempotent ALTER TABLE for columns added after the initial schema.

        SQLAlchemy's `create_all` does not modify existing tables, so any
        column added later must be back-filled here for users with an
        existing local DB. Each statement is wrapped so that "duplicate
        column" / "no such table" errors are silently ignored — both are
        expected on already-migrated or fresh databases respectively.
        """
        statements = [
            "ALTER TABLE accountability_reports "
            "ADD COLUMN root_cause_reason TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE accountability_reports "
            "ADD COLUMN memory_usage_json TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE accountability_reports "
            "ADD COLUMN evidence_used_by_final_answer_json TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE accountability_reports "
            "ADD COLUMN most_supportive_evidence_ids_json TEXT NOT NULL DEFAULT '[]'",
        ]
        for ddl in statements:
            try:
                with self._engine.connect() as conn:
                    conn.execute(text(ddl))
                    conn.commit()
            except Exception:
                # Column already exists, table not yet created, or backend
                # doesn't support ALTER — all benign here.
                pass

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
            row.root_cause_reason                = report.root_cause_reason
            row.memory_usage_json                = _dumps(
                [u.to_dict() for u in report.memory_usage]
            )
            row.evidence_used_by_final_answer_json = _dumps(
                list(report.evidence_used_by_final_answer)
            )
            row.most_supportive_evidence_ids_json = _dumps(
                list(report.most_supportive_evidence_ids)
            )
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

    # ------------------------------------------------------------------
    # Manual reviews (v2 — ORM-managed, FK-linked)
    # ------------------------------------------------------------------

    def latest_pipeline_task_id_for(self, medqa_task_id: str) -> Optional[str]:
        """
        Find the most recent ``tasks.task_id`` whose
        ``input.raw_task_id`` matches ``medqa_task_id``.

        Returns None when no pipeline run exists for that MedQA record.
        Used by `save_manual_review` to populate the soft FK.
        """
        if not medqa_task_id:
            return None
        with self._Session() as session:
            rows = (
                session.query(_Task)
                .order_by(_Task.created_at.desc())
                .all()
            )
            for r in rows:
                inp = _loads(r.input_json) or {}
                if str(inp.get("raw_task_id", "")) == str(medqa_task_id):
                    return r.task_id
        return None

    def save_manual_review(
        self,
        *,
        medqa_task_id: str,
        plausibility: Optional[int] = None,
        completeness: Optional[int] = None,
        specificity: Optional[int] = None,
        causal_coherence: Optional[int] = None,
        notes: str = "",
        status: str = "reviewed",
        pipeline_task_id: Optional[str] = None,
        link_pipeline_task: bool = True,
        reviewed_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upsert a manual review row keyed by ``medqa_task_id``.

        When ``pipeline_task_id`` is None and ``link_pipeline_task`` is
        True (the default), the most recent pipeline run for this
        ``medqa_task_id`` is looked up and linked. Pass an explicit
        ``pipeline_task_id`` to lock the link to a specific run, or
        ``link_pipeline_task=False`` to save without any link.

        The FK to ``tasks.task_id`` is enforced at write time: if
        ``pipeline_task_id`` is provided but doesn't exist in tasks, the
        write raises ``KeyError`` rather than silently storing an orphan.
        """
        if not medqa_task_id:
            raise ValueError("medqa_task_id is required")

        if pipeline_task_id is None and link_pipeline_task:
            pipeline_task_id = self.latest_pipeline_task_id_for(medqa_task_id)

        # Validate the FK ourselves — SQLite doesn't enforce FKs unless
        # PRAGMA foreign_keys=ON, and even then only for declared targets.
        if pipeline_task_id:
            with self._Session() as session:
                if session.get(_Task, pipeline_task_id) is None:
                    raise KeyError(
                        f"pipeline_task_id={pipeline_task_id!r} does not "
                        "exist in tasks; refusing to create orphan review."
                    )

        ts = reviewed_at or datetime.now(timezone.utc).isoformat()

        with self._Session() as session:
            row = (
                session.query(_ManualReview)
                .filter_by(medqa_task_id=medqa_task_id)
                .one_or_none()
            )
            if row is None:
                row = _ManualReview(medqa_task_id=medqa_task_id)
                session.add(row)
            row.pipeline_task_id = pipeline_task_id
            row.plausibility     = plausibility
            row.completeness     = completeness
            row.specificity      = specificity
            row.causal_coherence = causal_coherence
            row.notes            = notes or ""
            row.status           = status
            row.reviewed_at      = ts
            session.commit()
            return _manual_review_to_dict(row)

    def get_manual_review(self, medqa_task_id: str) -> Optional[Dict[str, Any]]:
        with self._Session() as session:
            row = (
                session.query(_ManualReview)
                .filter_by(medqa_task_id=medqa_task_id)
                .one_or_none()
            )
            return _manual_review_to_dict(row) if row else None

    def list_manual_reviews(
        self,
        status: Optional[str] = None,
        include_legacy: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        List manual reviews from the v2 table.

        When ``include_legacy`` is True, also reads from the legacy raw-SQL
        ``manual_reviews`` table and returns rows whose ``task_id`` does
        NOT already appear in v2 (so v2 takes precedence on overlap). The
        legacy rows are normalised to the v2 dict shape so downstream
        consumers see one uniform stream.
        """
        out: List[Dict[str, Any]] = []
        seen_medqa: set = set()

        with self._Session() as session:
            q = session.query(_ManualReview)
            if status is not None:
                q = q.filter_by(status=status)
            for row in q.order_by(_ManualReview.reviewed_at).all():
                d = _manual_review_to_dict(row)
                seen_medqa.add(d["medqa_task_id"])
                out.append(d)

        if include_legacy:
            for d in self._read_legacy_manual_reviews(status=status):
                if d["medqa_task_id"] in seen_medqa:
                    continue
                out.append(d)

        return out

    def _read_legacy_manual_reviews(
        self, status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Read the legacy `manual_reviews` table if it exists. Empty otherwise."""
        sql = "SELECT * FROM manual_reviews"
        params: Dict[str, Any] = {}
        if status is not None:
            sql += " WHERE status = :status"
            params["status"] = status
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(text(sql), params).mappings().all()
        except Exception:
            return []
        return [
            {
                "review_id":        r.get("review_id"),
                "medqa_task_id":    r.get("task_id"),
                "pipeline_task_id": None,   # legacy rows pre-date the FK
                "plausibility":     r.get("plausibility"),
                "completeness":     r.get("completeness"),
                "specificity":      r.get("specificity"),
                "causal_coherence": r.get("causal_coherence"),
                "notes":            r.get("notes") or "",
                "status":           r.get("status") or "",
                "reviewed_at":      r.get("reviewed_at") or "",
                "source":           "legacy",
            }
            for r in rows
        ]

    def migrate_legacy_manual_reviews(self) -> int:
        """
        Copy rows from legacy ``manual_reviews`` into ``manual_reviews_v2``.

        Idempotent: a v2 row keyed by the same ``medqa_task_id`` is left
        untouched (v2 always wins on conflict). Returns the number of
        rows actually copied. Safe to call on every store init —
        ``trajectory_store`` runs it from ``_run_lightweight_migrations``.
        """
        legacy = self._read_legacy_manual_reviews()
        if not legacy:
            return 0
        copied = 0
        with self._Session() as session:
            existing = {
                r.medqa_task_id
                for r in session.query(_ManualReview.medqa_task_id).all()
            }
            for row in legacy:
                medqa_id = row.get("medqa_task_id")
                if not medqa_id or medqa_id in existing:
                    continue
                session.add(_ManualReview(
                    medqa_task_id=medqa_id,
                    pipeline_task_id=None,
                    plausibility=row.get("plausibility"),
                    completeness=row.get("completeness"),
                    specificity=row.get("specificity"),
                    causal_coherence=row.get("causal_coherence"),
                    notes=row.get("notes") or "",
                    status=row.get("status") or "reviewed",
                    reviewed_at=row.get("reviewed_at") or "",
                ))
                copied += 1
            session.commit()
        return copied

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
