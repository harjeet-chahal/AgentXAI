"""
Tests for ``agentxai.api.main``.

Uses FastAPI's TestClient against a pre-seeded in-memory SQLite TrajectoryStore
injected via dependency overrides. The ``run_task`` pipeline is replaced with a
stub that writes a new task row to the same store, so POST /tasks/run is
testable without invoking any LLMs or real tools.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from agentxai.api.main import app, get_pipeline, get_store
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
from agentxai.store.trajectory_store import TrajectoryStore


TASK_ID = "API-TEST-001"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _seeded_record() -> AgentXAIRecord:
    event = TrajectoryEvent(
        event_id="evt-001",
        agent_id="orchestrator",
        event_type="plan",
        state_before={"step": 0},
        action="emit_plan",
        action_inputs={"case_id": TASK_ID},
        state_after={"step": 1},
        outcome="success",
    )
    plan = AgentPlan(
        plan_id="plan-001",
        agent_id="orchestrator",
        intended_actions=["route_to_a", "route_to_b", "synthesize"],
        actual_actions=["route_to_a", "synthesize"],
        deviations=["route_to_b"],
        deviation_reasons=["specialist_b timed out"],
    )
    tool = ToolUseEvent(
        tool_call_id="tool-001",
        tool_name="symptom_lookup",
        called_by="specialist_a",
        inputs={"symptom": "chest_pain"},
        outputs={"conditions": ["MI"]},
        duration_ms=12.5,
        downstream_impact_score=0.75,
    )
    diff = MemoryDiff(
        diff_id="diff-001",
        agent_id="specialist_a",
        operation="write",
        key="severity",
        value_before=None,
        value_after=0.9,
        triggered_by_event_id=event.event_id,
    )
    msg = AgentMessage(
        message_id="msg-001",
        sender="specialist_a",
        receiver="orchestrator",
        message_type="finding",
        content={"top": "MI"},
        acted_upon=True,
        behavior_change_description="Orchestrator escalated to Synthesizer.",
    )
    edge = CausalEdge(
        edge_id="edge-001",
        cause_event_id=event.event_id,
        effect_event_id="evt-downstream",
        causal_strength=0.82,
        causal_type="direct",
    )
    report = AccountabilityReport(
        task_id=TASK_ID,
        final_outcome="MI",
        outcome_correct=True,
        agent_responsibility_scores={"specialist_a": 0.8, "synthesizer": 0.2},
        root_cause_event_id=event.event_id,
        causal_chain=[event.event_id, "evt-downstream"],
        most_impactful_tool_call_id=tool.tool_call_id,
        critical_memory_diffs=[diff.diff_id],
        most_influential_message_id=msg.message_id,
        plan_deviation_summary="specialist_b was skipped",
        one_line_explanation="Symptom analysis drove the MI diagnosis.",
    )
    return AgentXAIRecord(
        task_id=TASK_ID,
        source="medqa",
        input={"patient_case": "45yo chest pain", "options": {"A": "MI"}},
        ground_truth={"correct_answer": "A", "explanation": "ST elevation"},
        system_output={"final_diagnosis": "MI", "confidence": 0.88, "correct": True},
        xai_data=XAIData(
            trajectory=[event],
            plans=[plan],
            tool_calls=[tool],
            memory_diffs=[diff],
            messages=[msg],
            causal_graph=CausalGraph(
                nodes=[event.event_id, "evt-downstream"],
                edges=[edge],
            ),
            accountability_report=report,
        ),
    )


def _save_all(store: TrajectoryStore, record: AgentXAIRecord) -> None:
    store.save_task(record)
    for ev in record.xai_data.trajectory:
        store.save_event(record.task_id, ev)
    for p in record.xai_data.plans:
        store.save_plan(record.task_id, p)
    for tc in record.xai_data.tool_calls:
        store.save_tool_call(record.task_id, tc)
    for d in record.xai_data.memory_diffs:
        store.save_memory_diff(record.task_id, d)
    for m in record.xai_data.messages:
        store.save_message(record.task_id, m)
    for e in record.xai_data.causal_graph.edges:
        store.save_causal_edge(record.task_id, e)
    if record.xai_data.accountability_report is not None:
        store.save_accountability_report(record.xai_data.accountability_report)


class _StubPipeline:
    """Minimal Pipeline replacement: persists a new task row and returns its record."""

    def __init__(self, store: TrajectoryStore) -> None:
        self.store = store
        self.calls: list[Dict[str, Any]] = []

    def run_task(self, record: Dict[str, Any]) -> AgentXAIRecord:
        self.calls.append(record)
        new_id = f"API-TEST-NEW-{len(self.calls):03d}"
        new_record = AgentXAIRecord(
            task_id=new_id,
            source="medqa",
            input={"patient_case": record.get("question", "")},
            ground_truth={"correct_answer": record.get("answer", "")},
            system_output={"final_diagnosis": "stub", "correct": False},
        )
        self.store.save_task(new_record)
        return new_record


@pytest.fixture()
def store(tmp_path) -> TrajectoryStore:
    # File-backed (not :memory:) because FastAPI's TestClient dispatches sync
    # handlers onto a threadpool, and SQLAlchemy's default SQLite pool gives
    # each thread its own connection — which, for :memory:, means an empty DB.
    s = TrajectoryStore(db_url=f"sqlite:///{tmp_path/'api.db'}")
    _save_all(s, _seeded_record())
    return s


@pytest.fixture()
def pipeline(store: TrajectoryStore) -> _StubPipeline:
    return _StubPipeline(store)


@pytest.fixture()
def client(store: TrajectoryStore, pipeline: _StubPipeline) -> TestClient:
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /tasks
# ---------------------------------------------------------------------------

class TestListTasks:
    def test_lists_seeded_task(self, client):
        r = client.get("/tasks")
        assert r.status_code == 200
        body = r.json()
        assert body["page"] == 1
        assert body["per_page"] == 50
        assert body["total"] == 1
        assert len(body["items"]) == 1
        item = body["items"][0]
        assert item["task_id"] == TASK_ID
        assert item["source"] == "medqa"
        assert item["outcome_correct"] is True
        assert item["final_outcome"] == "MI"
        assert item["created_at"] is not None

    def test_paginates_by_50(self, client, store):
        for i in range(55):
            store.save_task(AgentXAIRecord(task_id=f"T-{i:03d}"))
        r1 = client.get("/tasks?page=1")
        r2 = client.get("/tasks?page=2")
        assert r1.status_code == 200 and r2.status_code == 200
        body1, body2 = r1.json(), r2.json()
        assert body1["total"] == 56
        assert body2["total"] == 56
        assert len(body1["items"]) == 50
        assert len(body2["items"]) == 6
        ids_page1 = {i["task_id"] for i in body1["items"]}
        ids_page2 = {i["task_id"] for i in body2["items"]}
        assert ids_page1.isdisjoint(ids_page2)
        assert TASK_ID in (ids_page1 | ids_page2)

    def test_respects_custom_per_page(self, client, store):
        for i in range(4):
            store.save_task(AgentXAIRecord(task_id=f"P-{i}"))
        r = client.get("/tasks?page=2&per_page=2")
        assert r.status_code == 200
        body = r.json()
        assert body["per_page"] == 2
        assert body["total"] == 5
        assert len(body["items"]) == 2

    def test_rejects_bad_pagination(self, client):
        assert client.get("/tasks?page=0").status_code == 422
        assert client.get("/tasks?per_page=0").status_code == 422


# ---------------------------------------------------------------------------
# GET /tasks/{task_id}
# ---------------------------------------------------------------------------

class TestGetTask:
    def test_returns_full_record(self, client):
        r = client.get(f"/tasks/{TASK_ID}")
        assert r.status_code == 200
        body = r.json()
        assert body["task_id"] == TASK_ID
        assert body["source"] == "medqa"
        assert body["input"]["patient_case"].startswith("45yo")
        assert body["ground_truth"]["correct_answer"] == "A"
        assert body["system_output"]["final_diagnosis"] == "MI"
        xai = body["xai_data"]
        assert len(xai["trajectory"]) == 1
        assert len(xai["plans"]) == 1
        assert len(xai["tool_calls"]) == 1
        assert len(xai["memory_diffs"]) == 1
        assert len(xai["messages"]) == 1
        assert len(xai["causal_graph"]["edges"]) == 1
        assert xai["accountability_report"]["final_outcome"] == "MI"

    def test_404_when_missing(self, client):
        r = client.get("/tasks/DOES-NOT-EXIST")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Per-pillar endpoints
# ---------------------------------------------------------------------------

class TestPillarEndpoints:
    def test_trajectory(self, client):
        r = client.get(f"/tasks/{TASK_ID}/trajectory")
        assert r.status_code == 200
        events = r.json()
        assert len(events) == 1
        e = events[0]
        assert e["event_id"] == "evt-001"
        assert e["agent_id"] == "orchestrator"
        assert e["event_type"] == "plan"
        assert e["state_after"] == {"step": 1}

    def test_plans(self, client):
        r = client.get(f"/tasks/{TASK_ID}/plans")
        assert r.status_code == 200
        plans = r.json()
        assert len(plans) == 1
        p = plans[0]
        assert p["intended_actions"] == ["route_to_a", "route_to_b", "synthesize"]
        assert p["deviations"] == ["route_to_b"]

    def test_tools(self, client):
        r = client.get(f"/tasks/{TASK_ID}/tools")
        assert r.status_code == 200
        tools = r.json()
        assert len(tools) == 1
        t = tools[0]
        assert t["tool_name"] == "symptom_lookup"
        assert t["called_by"] == "specialist_a"
        assert t["downstream_impact_score"] == pytest.approx(0.75)

    def test_memory(self, client):
        r = client.get(f"/tasks/{TASK_ID}/memory")
        assert r.status_code == 200
        diffs = r.json()
        assert len(diffs) == 1
        d = diffs[0]
        assert d["operation"] == "write"
        assert d["key"] == "severity"
        assert d["value_after"] == 0.9

    def test_messages(self, client):
        r = client.get(f"/tasks/{TASK_ID}/messages")
        assert r.status_code == 200
        msgs = r.json()
        assert len(msgs) == 1
        m = msgs[0]
        assert m["sender"] == "specialist_a"
        assert m["receiver"] == "orchestrator"
        assert m["acted_upon"] is True

    def test_causal(self, client):
        r = client.get(f"/tasks/{TASK_ID}/causal")
        assert r.status_code == 200
        graph = r.json()
        assert set(graph.keys()) == {"nodes", "edges"}
        assert len(graph["edges"]) == 1
        e = graph["edges"][0]
        assert e["cause_event_id"] == "evt-001"
        assert e["effect_event_id"] == "evt-downstream"
        assert e["causal_type"] == "direct"
        assert "evt-001" in graph["nodes"]
        assert "evt-downstream" in graph["nodes"]

    def test_accountability(self, client):
        r = client.get(f"/tasks/{TASK_ID}/accountability")
        assert r.status_code == 200
        ar = r.json()
        assert ar["final_outcome"] == "MI"
        assert ar["outcome_correct"] is True
        assert ar["agent_responsibility_scores"]["specialist_a"] == pytest.approx(0.8)
        assert ar["causal_chain"] == ["evt-001", "evt-downstream"]

    def test_pillar_404_when_task_missing(self, client):
        for suffix in (
            "trajectory", "plans", "tools", "memory",
            "messages", "causal", "accountability",
        ):
            r = client.get(f"/tasks/DOES-NOT-EXIST/{suffix}")
            assert r.status_code == 404, f"{suffix} should 404"

    def test_accountability_404_when_report_missing(self, client, store):
        bare_id = "BARE-001"
        store.save_task(AgentXAIRecord(task_id=bare_id))
        r = client.get(f"/tasks/{bare_id}/accountability")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /tasks/run
# ---------------------------------------------------------------------------

class TestRunTask:
    def test_invokes_pipeline_and_returns_task_id(self, client, pipeline):
        payload = {
            "record": {
                "question": "62yo with crushing chest pain",
                "options": {"A": "MI", "B": "PE", "C": "GERD", "D": "anxiety"},
                "answer": "A",
            }
        }
        r = client.post("/tasks/run", json=payload)
        assert r.status_code == 200
        new_id = r.json()["task_id"]
        assert new_id.startswith("API-TEST-NEW-")
        assert pipeline.calls == [payload["record"]]

        # Persisted task is fetchable via the GET endpoints.
        follow = client.get(f"/tasks/{new_id}")
        assert follow.status_code == 200
        assert follow.json()["task_id"] == new_id

    def test_rejects_missing_record(self, client):
        r = client.post("/tasks/run", json={})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

class TestCORS:
    def test_allows_streamlit_origin(self, client):
        r = client.get("/tasks", headers={"Origin": "http://localhost:8501"})
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "*"

    def test_preflight_allows_any_method(self, client):
        r = client.options(
            "/tasks/run",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "*"
        assert "POST" in r.headers.get("access-control-allow-methods", "")
