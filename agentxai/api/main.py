"""
FastAPI application entry point.

Exposes endpoints for running the pipeline and querying XAI data from the
trajectory store. Pydantic v2 response models mirror the dataclasses in
``agentxai.data.schemas`` one-for-one.

Security posture
----------------
The API is designed for **local-first** use (Streamlit dashboard + the
researcher's laptop). It supports two opt-in hardening levers via env
vars; defaults preserve the friction-free local dev experience.

* **CORS origins** — by default the API only accepts cross-origin
  requests from common localhost ports (Streamlit on 8501, Uvicorn on
  8000, etc.). Override with:

    - ``AGENTXAI_CORS_ORIGINS=https://my.app,https://other.app`` —
      explicit allow-list (comma-separated).
    - ``AGENTXAI_ALLOW_CORS_ALL=true`` — emergency wildcard, restoring
      the old ``["*"]`` behaviour. Only use when you genuinely need to
      accept any origin (e.g., a public demo); never with a non-empty
      ``AGENTXAI_API_TOKEN`` would-be-credentialled flow.

* **POST /tasks/run token** — disabled by default. Set
  ``AGENTXAI_API_TOKEN=<secret>`` and clients must send
  ``Authorization: Bearer <secret>`` on the run endpoint or get 401.
  Read endpoints stay unauthenticated — they only expose data the
  caller could see by reading the SQLite file directly.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from agentxai.data.schemas import AgentXAIRecord
from agentxai.store.trajectory_store import TrajectoryStore


# ---------------------------------------------------------------------------
# Pydantic v2 response models (mirror the dataclasses in data/schemas.py)
# ---------------------------------------------------------------------------

class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class TrajectoryEventModel(_Base):
    event_id: str
    timestamp: float
    agent_id: str = ""
    event_type: str = ""
    state_before: Dict[str, Any] = Field(default_factory=dict)
    action: str = ""
    action_inputs: Dict[str, Any] = Field(default_factory=dict)
    state_after: Dict[str, Any] = Field(default_factory=dict)
    outcome: str = ""


class AgentPlanModel(_Base):
    plan_id: str
    agent_id: str = ""
    timestamp: float
    intended_actions: List[str] = Field(default_factory=list)
    actual_actions: List[str] = Field(default_factory=list)
    deviations: List[str] = Field(default_factory=list)
    deviation_reasons: List[str] = Field(default_factory=list)


class ToolUseEventModel(_Base):
    tool_call_id: str
    tool_name: str = ""
    called_by: str = ""
    timestamp: float
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0
    downstream_impact_score: float = 0.0
    counterfactual_run_id: str = ""


class MemoryDiffModel(_Base):
    diff_id: str
    agent_id: str = ""
    timestamp: float
    operation: str = ""
    key: str = ""
    value_before: Any = None
    value_after: Any = None
    triggered_by_event_id: str = ""


class AgentMessageModel(_Base):
    message_id: str
    sender: str = ""
    receiver: str = ""
    timestamp: float
    message_type: str = ""
    content: Dict[str, Any] = Field(default_factory=dict)
    acted_upon: bool = False
    behavior_change_description: str = ""


class CausalEdgeModel(_Base):
    edge_id: str
    cause_event_id: str = ""
    effect_event_id: str = ""
    causal_strength: float = 0.0
    causal_type: str = ""


class CausalGraphModel(_Base):
    nodes: List[str] = Field(default_factory=list)
    edges: List[CausalEdgeModel] = Field(default_factory=list)


class AccountabilityReportModel(_Base):
    task_id: str
    final_outcome: str = ""
    outcome_correct: bool = False
    agent_responsibility_scores: Dict[str, float] = Field(default_factory=dict)
    root_cause_event_id: str = ""
    causal_chain: List[str] = Field(default_factory=list)
    most_impactful_tool_call_id: str = ""
    critical_memory_diffs: List[str] = Field(default_factory=list)
    most_influential_message_id: str = ""
    plan_deviation_summary: str = ""
    one_line_explanation: str = ""


class XAIDataModel(_Base):
    trajectory: List[TrajectoryEventModel] = Field(default_factory=list)
    plans: List[AgentPlanModel] = Field(default_factory=list)
    tool_calls: List[ToolUseEventModel] = Field(default_factory=list)
    memory_diffs: List[MemoryDiffModel] = Field(default_factory=list)
    messages: List[AgentMessageModel] = Field(default_factory=list)
    causal_graph: CausalGraphModel = Field(default_factory=CausalGraphModel)
    accountability_report: Optional[AccountabilityReportModel] = None


class AgentXAIRecordModel(_Base):
    task_id: str
    source: str = "medqa"
    input: Dict[str, Any] = Field(default_factory=dict)
    ground_truth: Dict[str, Any] = Field(default_factory=dict)
    system_output: Dict[str, Any] = Field(default_factory=dict)
    xai_data: XAIDataModel = Field(default_factory=XAIDataModel)


class TaskSummaryModel(_Base):
    task_id: str
    source: str
    created_at: Optional[str] = None
    final_outcome: Optional[str] = None
    outcome_correct: Optional[bool] = None


class TaskListResponse(_Base):
    items: List[TaskSummaryModel]
    page: int
    per_page: int
    total: int


class RunTaskRequest(_Base):
    record: Dict[str, Any]


class RunTaskResponse(_Base):
    task_id: str


# ---------------------------------------------------------------------------
# Dependencies (overridable in tests via app.dependency_overrides)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_store() -> TrajectoryStore:
    """Return the process-wide TrajectoryStore singleton (file-backed)."""
    return TrajectoryStore()


def get_pipeline() -> Any:
    """Return a fresh Pipeline. Imported lazily to avoid loading LLMs at startup."""
    from run_pipeline import Pipeline
    return Pipeline()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_record(store: TrajectoryStore, task_id: str) -> AgentXAIRecord:
    try:
        return store.get_full_record(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"task {task_id!r} not found")


def _paginate_tasks(
    store: TrajectoryStore,
    page: int,
    per_page: int,
) -> tuple[int, List[TaskSummaryModel]]:
    from agentxai.store.trajectory_store import _AccountabilityReport, _Task

    offset = (page - 1) * per_page
    with store._Session() as session:
        total = session.query(_Task).count()
        rows = (
            session.query(_Task)
            .order_by(_Task.created_at.desc())
            .offset(offset)
            .limit(per_page)
            .all()
        )
        items: List[TaskSummaryModel] = []
        for t in rows:
            ar = session.get(_AccountabilityReport, t.task_id)
            items.append(
                TaskSummaryModel(
                    task_id=t.task_id,
                    source=t.source,
                    created_at=t.created_at.isoformat() if t.created_at else None,
                    final_outcome=ar.final_outcome if ar else None,
                    outcome_correct=bool(ar.outcome_correct) if ar else None,
                )
            )
    return total, items


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="AgentXAI API", version="0.1.0")


# ---------------------------------------------------------------------------
# CORS — env-driven, localhost-only by default
# ---------------------------------------------------------------------------

# Built-in localhost allow-list. Covers the Streamlit dashboard (8501),
# the API itself (8000), and the equivalent loopback variants. Both
# 127.0.0.1 and localhost are listed because browsers treat them as
# distinct origins — same machine, different host strings.
_DEFAULT_LOCAL_ORIGINS: List[str] = [
    "http://localhost:8501", "http://127.0.0.1:8501",
    "http://localhost:8000", "http://127.0.0.1:8000",
    "http://localhost:3000", "http://127.0.0.1:3000",
]


def _resolve_cors_origins() -> List[str]:
    """
    Compute the CORS allow-list from env, in priority order:

      1. ``AGENTXAI_ALLOW_CORS_ALL=true`` → ``["*"]`` (emergency override).
      2. ``AGENTXAI_CORS_ORIGINS`` (comma-separated) → that list verbatim.
      3. Default localhost-only allow-list.

    Pulled into a function so tests can monkey-patch the env and rebuild
    the app without restarting the process.
    """
    if os.environ.get("AGENTXAI_ALLOW_CORS_ALL", "").lower() == "true":
        return ["*"]
    raw = os.environ.get("AGENTXAI_CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return list(_DEFAULT_LOCAL_ORIGINS)


app.add_middleware(
    CORSMiddleware,
    allow_origins=_resolve_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Optional API-token auth for write endpoints
# ---------------------------------------------------------------------------

# Token-validating dependency. When AGENTXAI_API_TOKEN is unset (the
# local-dev default) every call passes through; when set, requests must
# carry ``Authorization: Bearer <token>`` matching the env var.
def verify_api_token(
    authorization: Optional[str] = Header(default=None),
) -> None:
    """
    Validate the optional bearer token on write endpoints.

    Reads the expected token from ``AGENTXAI_API_TOKEN`` at request time
    so tests (and ops) can rotate it without restarting the app. When
    the env var is unset/empty the dependency is a no-op — the friction
    of token configuration is reserved for users who actually want auth.

    Raises ``HTTPException(401)`` on missing/malformed/wrong token.
    """
    expected = (os.environ.get("AGENTXAI_API_TOKEN") or "").strip()
    if not expected:
        return  # no token configured → no auth required (local dev)

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. "
                   "Set 'Authorization: Bearer <token>' to call write endpoints.",
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=401,
            detail="Authorization header must be 'Bearer <token>'.",
        )
    if token.strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid API token.")


@app.get("/tasks", response_model=TaskListResponse)
def list_tasks(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=500),
    store: TrajectoryStore = Depends(get_store),
) -> TaskListResponse:
    total, items = _paginate_tasks(store, page, per_page)
    return TaskListResponse(items=items, page=page, per_page=per_page, total=total)


@app.get("/tasks/{task_id}", response_model=AgentXAIRecordModel)
def get_task(
    task_id: str,
    store: TrajectoryStore = Depends(get_store),
) -> AgentXAIRecordModel:
    return AgentXAIRecordModel.model_validate(_require_record(store, task_id).to_dict())


@app.get("/tasks/{task_id}/trajectory", response_model=List[TrajectoryEventModel])
def get_trajectory(
    task_id: str,
    store: TrajectoryStore = Depends(get_store),
) -> List[TrajectoryEventModel]:
    rec = _require_record(store, task_id)
    return [TrajectoryEventModel.model_validate(e.to_dict()) for e in rec.xai_data.trajectory]


@app.get("/tasks/{task_id}/plans", response_model=List[AgentPlanModel])
def get_plans(
    task_id: str,
    store: TrajectoryStore = Depends(get_store),
) -> List[AgentPlanModel]:
    rec = _require_record(store, task_id)
    return [AgentPlanModel.model_validate(p.to_dict()) for p in rec.xai_data.plans]


@app.get("/tasks/{task_id}/tools", response_model=List[ToolUseEventModel])
def get_tools(
    task_id: str,
    store: TrajectoryStore = Depends(get_store),
) -> List[ToolUseEventModel]:
    rec = _require_record(store, task_id)
    return [ToolUseEventModel.model_validate(t.to_dict()) for t in rec.xai_data.tool_calls]


@app.get("/tasks/{task_id}/memory", response_model=List[MemoryDiffModel])
def get_memory(
    task_id: str,
    store: TrajectoryStore = Depends(get_store),
) -> List[MemoryDiffModel]:
    rec = _require_record(store, task_id)
    return [MemoryDiffModel.model_validate(m.to_dict()) for m in rec.xai_data.memory_diffs]


@app.get("/tasks/{task_id}/messages", response_model=List[AgentMessageModel])
def get_messages(
    task_id: str,
    store: TrajectoryStore = Depends(get_store),
) -> List[AgentMessageModel]:
    rec = _require_record(store, task_id)
    return [AgentMessageModel.model_validate(m.to_dict()) for m in rec.xai_data.messages]


@app.get("/tasks/{task_id}/causal", response_model=CausalGraphModel)
def get_causal(
    task_id: str,
    store: TrajectoryStore = Depends(get_store),
) -> CausalGraphModel:
    rec = _require_record(store, task_id)
    return CausalGraphModel.model_validate(rec.xai_data.causal_graph.to_dict())


@app.get("/tasks/{task_id}/accountability", response_model=AccountabilityReportModel)
def get_accountability(
    task_id: str,
    store: TrajectoryStore = Depends(get_store),
) -> AccountabilityReportModel:
    rec = _require_record(store, task_id)
    if rec.xai_data.accountability_report is None:
        raise HTTPException(
            status_code=404,
            detail=f"no accountability report for task {task_id!r}",
        )
    return AccountabilityReportModel.model_validate(
        rec.xai_data.accountability_report.to_dict()
    )


@app.post("/tasks/run", response_model=RunTaskResponse)
def run_task(
    req: RunTaskRequest,
    pipeline: Any = Depends(get_pipeline),
    _auth: None = Depends(verify_api_token),
) -> RunTaskResponse:
    """
    Execute the full pipeline against `req.record` and return the new task_id.

    Token-protected when ``AGENTXAI_API_TOKEN`` is set in the server's
    environment; unauthenticated otherwise (default local-dev mode).
    """
    result = pipeline.run_task(req.record)
    return RunTaskResponse(task_id=result.task_id)
