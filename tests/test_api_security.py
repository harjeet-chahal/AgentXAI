"""
Tests for the env-driven CORS allow-list and the optional API-token
auth on POST /tasks/run.

Two design constraints to keep in mind:
  * CORS middleware is registered when the app starts, so changing
    `_resolve_cors_origins`'s output mid-test requires rebuilding the
    middleware list directly on `app.user_middleware`.
  * `verify_api_token` reads `AGENTXAI_API_TOKEN` *at request time*, so
    flipping the env var with monkeypatch is enough — no app restart.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from agentxai.api.main import (
    _DEFAULT_LOCAL_ORIGINS,
    _resolve_cors_origins,
    app,
    get_pipeline,
    get_store,
    verify_api_token,
)
from agentxai.data.schemas import AgentXAIRecord
from agentxai.store.trajectory_store import TrajectoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _StubPipeline:
    """Minimal pipeline that just persists a fresh task and returns it."""

    def __init__(self, store: TrajectoryStore):
        self.store = store
        self._counter = 0

    def run_task(self, record: Dict[str, Any]) -> AgentXAIRecord:
        self._counter += 1
        rec = AgentXAIRecord(
            task_id=f"AUTH-TASK-{self._counter:03d}",
            source="test",
            input=dict(record or {}),
            ground_truth={},
            system_output={"final_diagnosis": "X"},
        )
        self.store.save_task(rec)
        return rec


@pytest.fixture()
def store(tmp_path) -> TrajectoryStore:
    return TrajectoryStore(db_url=f"sqlite:///{tmp_path / 'auth.db'}")


@pytest.fixture()
def client(store: TrajectoryStore) -> TestClient:
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_pipeline] = lambda: _StubPipeline(store)
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """
    Each test starts with no security-related env vars set so prior
    tests can't leak state. Tests that want auth on opt back in.
    """
    for var in (
        "AGENTXAI_API_TOKEN",
        "AGENTXAI_CORS_ORIGINS",
        "AGENTXAI_ALLOW_CORS_ALL",
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# CORS — env resolution
# ---------------------------------------------------------------------------

class TestResolveCorsOrigins:
    def test_default_is_localhost_only(self):
        # No env vars set → built-in localhost allow-list, NOT wildcard.
        origins = _resolve_cors_origins()
        assert "*" not in origins
        # Sanity: includes the standard Streamlit + Uvicorn ports.
        for expected in (
            "http://localhost:8501",
            "http://127.0.0.1:8501",
            "http://localhost:8000",
        ):
            assert expected in origins
        assert origins == list(_DEFAULT_LOCAL_ORIGINS)

    def test_wildcard_only_with_explicit_opt_in(self, monkeypatch):
        monkeypatch.setenv("AGENTXAI_ALLOW_CORS_ALL", "true")
        assert _resolve_cors_origins() == ["*"]

    def test_wildcard_opt_in_is_case_insensitive(self, monkeypatch):
        for value in ("true", "TRUE", "True"):
            monkeypatch.setenv("AGENTXAI_ALLOW_CORS_ALL", value)
            assert _resolve_cors_origins() == ["*"]

    def test_wildcard_off_for_falsy_values(self, monkeypatch):
        # Anything other than the literal "true" (case-insensitive)
        # leaves the localhost default in place — defensive against
        # typos like "1", "yes".
        for value in ("1", "yes", "false", "no", ""):
            monkeypatch.setenv("AGENTXAI_ALLOW_CORS_ALL", value)
            origins = _resolve_cors_origins()
            assert origins != ["*"], (
                f"AGENTXAI_ALLOW_CORS_ALL={value!r} should NOT enable wildcard"
            )

    def test_explicit_origins_override_default(self, monkeypatch):
        monkeypatch.setenv(
            "AGENTXAI_CORS_ORIGINS",
            "https://app.example.com,https://other.example.com",
        )
        origins = _resolve_cors_origins()
        assert origins == ["https://app.example.com", "https://other.example.com"]

    def test_explicit_origins_strip_whitespace_and_empties(self, monkeypatch):
        monkeypatch.setenv(
            "AGENTXAI_CORS_ORIGINS",
            "  https://a.example.com , , https://b.example.com ",
        )
        origins = _resolve_cors_origins()
        assert origins == ["https://a.example.com", "https://b.example.com"]

    def test_wildcard_takes_priority_over_explicit_origins(self, monkeypatch):
        # If the operator set both, wildcard wins (it's the explicit
        # opt-in). Documents the precedence so it isn't surprising.
        monkeypatch.setenv("AGENTXAI_ALLOW_CORS_ALL", "true")
        monkeypatch.setenv(
            "AGENTXAI_CORS_ORIGINS", "https://only.example.com",
        )
        assert _resolve_cors_origins() == ["*"]


# ---------------------------------------------------------------------------
# verify_api_token (the dependency in isolation)
# ---------------------------------------------------------------------------

class TestVerifyApiTokenUnit:
    def test_no_op_when_token_unset(self):
        # Empty env var → no-op. Should not raise even with no header.
        assert verify_api_token(authorization=None) is None

    def test_no_op_when_token_empty_string(self, monkeypatch):
        monkeypatch.setenv("AGENTXAI_API_TOKEN", "")
        assert verify_api_token(authorization=None) is None

    def test_raises_when_token_set_but_header_missing(self, monkeypatch):
        monkeypatch.setenv("AGENTXAI_API_TOKEN", "secret")
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            verify_api_token(authorization=None)
        assert exc_info.value.status_code == 401

    def test_raises_on_wrong_scheme(self, monkeypatch):
        monkeypatch.setenv("AGENTXAI_API_TOKEN", "secret")
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            verify_api_token(authorization="Basic abcdef")

    def test_raises_on_wrong_token(self, monkeypatch):
        monkeypatch.setenv("AGENTXAI_API_TOKEN", "secret")
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            verify_api_token(authorization="Bearer wrong")

    def test_passes_on_correct_token(self, monkeypatch):
        monkeypatch.setenv("AGENTXAI_API_TOKEN", "secret")
        # No exception means pass.
        assert verify_api_token(authorization="Bearer secret") is None


# ---------------------------------------------------------------------------
# POST /tasks/run — end-to-end via TestClient
# ---------------------------------------------------------------------------

# Minimal valid payload — the stub pipeline doesn't care about contents.
_VALID_RUN_PAYLOAD = {
    "record": {"question": "x", "options": {"A": "y"}, "answer_idx": 0}
}


class TestTasksRunNoToken:
    def test_no_token_required_when_unset(self, client):
        # Default local-dev mode: every call goes through, no auth needed.
        r = client.post("/tasks/run", json=_VALID_RUN_PAYLOAD)
        assert r.status_code == 200, r.text
        assert "task_id" in r.json()


class TestTasksRunWithToken:
    def test_request_with_correct_token_succeeds(self, client, monkeypatch):
        monkeypatch.setenv("AGENTXAI_API_TOKEN", "shh")
        r = client.post(
            "/tasks/run",
            json=_VALID_RUN_PAYLOAD,
            headers={"Authorization": "Bearer shh"},
        )
        assert r.status_code == 200, r.text
        assert "task_id" in r.json()

    def test_missing_header_returns_401(self, client, monkeypatch):
        monkeypatch.setenv("AGENTXAI_API_TOKEN", "shh")
        r = client.post("/tasks/run", json=_VALID_RUN_PAYLOAD)
        assert r.status_code == 401
        assert "Authorization" in r.json().get("detail", "")

    def test_wrong_scheme_returns_401(self, client, monkeypatch):
        monkeypatch.setenv("AGENTXAI_API_TOKEN", "shh")
        r = client.post(
            "/tasks/run",
            json=_VALID_RUN_PAYLOAD,
            headers={"Authorization": "Basic shh"},
        )
        assert r.status_code == 401

    def test_wrong_token_returns_401(self, client, monkeypatch):
        monkeypatch.setenv("AGENTXAI_API_TOKEN", "shh")
        r = client.post(
            "/tasks/run",
            json=_VALID_RUN_PAYLOAD,
            headers={"Authorization": "Bearer not-the-right-token"},
        )
        assert r.status_code == 401
        assert "Invalid" in r.json().get("detail", "")

    def test_read_endpoints_remain_unauthenticated_when_token_set(
        self, client, monkeypatch, store,
    ):
        # Token gates POST /tasks/run only; read endpoints stay open
        # because they expose what's already in the SQLite file.
        monkeypatch.setenv("AGENTXAI_API_TOKEN", "shh")
        # Seed a task so /tasks has something to return.
        store.save_task(AgentXAIRecord(task_id="READ-OK", source="test"))
        r = client.get("/tasks")
        assert r.status_code == 200
        assert any(item["task_id"] == "READ-OK" for item in r.json()["items"])

    def test_token_can_rotate_at_request_time(self, client, monkeypatch):
        # The dependency reads AGENTXAI_API_TOKEN at request time, so an
        # ops rotation takes effect without a server restart.
        monkeypatch.setenv("AGENTXAI_API_TOKEN", "old-secret")
        r1 = client.post(
            "/tasks/run", json=_VALID_RUN_PAYLOAD,
            headers={"Authorization": "Bearer old-secret"},
        )
        assert r1.status_code == 200

        monkeypatch.setenv("AGENTXAI_API_TOKEN", "new-secret")
        # Old token now rejected.
        r2 = client.post(
            "/tasks/run", json=_VALID_RUN_PAYLOAD,
            headers={"Authorization": "Bearer old-secret"},
        )
        assert r2.status_code == 401
        # New token accepted.
        r3 = client.post(
            "/tasks/run", json=_VALID_RUN_PAYLOAD,
            headers={"Authorization": "Bearer new-secret"},
        )
        assert r3.status_code == 200


# ---------------------------------------------------------------------------
# Dashboard client header passthrough
# ---------------------------------------------------------------------------

class TestDashboardApiHeaders:
    def test_no_authorization_header_when_token_unset(self, monkeypatch):
        from agentxai.ui.dashboard import _api_headers
        # Defensive: clean both names just in case.
        monkeypatch.delenv("AGENTXAI_API_TOKEN", raising=False)
        assert _api_headers() == {}

    def test_authorization_header_when_token_set(self, monkeypatch):
        from agentxai.ui.dashboard import _api_headers
        monkeypatch.setenv("AGENTXAI_API_TOKEN", "shh")
        assert _api_headers() == {"Authorization": "Bearer shh"}

    def test_blank_token_treated_as_unset(self, monkeypatch):
        from agentxai.ui.dashboard import _api_headers
        # `export AGENTXAI_API_TOKEN=` (empty value) shouldn't send a
        # malformed `Authorization: Bearer ` header.
        monkeypatch.setenv("AGENTXAI_API_TOKEN", "   ")
        assert _api_headers() == {}
