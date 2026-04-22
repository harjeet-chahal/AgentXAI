"""
Tests for agentxai/xai/memory_logger.py — Pillar 4.
"""

from __future__ import annotations

import pytest

from agentxai.data.schemas import AgentXAIRecord
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.memory_logger import LoggedMemory, MemoryLogger, current_event_id


TASK_ID = "MEM-TEST-001"


@pytest.fixture()
def store() -> TrajectoryStore:
    s = TrajectoryStore(db_url="sqlite:///:memory:")
    s.save_task(AgentXAIRecord(task_id=TASK_ID, source="test"))
    return s


@pytest.fixture()
def mem_logger(store: TrajectoryStore) -> MemoryLogger:
    return MemoryLogger(store=store, task_id=TASK_ID)


# ---------------------------------------------------------------------------
# Core: 3 writes + 2 reads ⇒ 5 diffs
# ---------------------------------------------------------------------------

class TestThreeWritesTwoReads:
    def test_five_diffs_with_correct_values_and_agent(self, mem_logger, store):
        mem = mem_logger.for_agent("specialist_a")

        mem["severity"] = 0.8      # write 1: before=None, after=0.8
        mem["diagnosis"] = "MI"    # write 2: before=None, after="MI"
        mem["severity"] = 0.9      # write 3: before=0.8, after=0.9

        assert mem["severity"] == 0.9    # read 1
        assert mem["diagnosis"] == "MI"  # read 2

        diffs = store.get_full_record(TASK_ID).xai_data.memory_diffs
        assert len(diffs) == 5

        # All diffs belong to the same agent
        for d in diffs:
            assert d.agent_id == "specialist_a"

        writes = [d for d in diffs if d.operation == "write"]
        reads = [d for d in diffs if d.operation == "read"]
        assert len(writes) == 3
        assert len(reads) == 2

        # Writes, in order
        assert writes[0].key == "severity"
        assert writes[0].value_before is None
        assert writes[0].value_after == 0.8

        assert writes[1].key == "diagnosis"
        assert writes[1].value_before is None
        assert writes[1].value_after == "MI"

        assert writes[2].key == "severity"
        assert writes[2].value_before == 0.8
        assert writes[2].value_after == 0.9

        # Reads capture value_before == value_after == current_value
        assert reads[0].key == "severity"
        assert reads[0].value_before == 0.9
        assert reads[0].value_after == 0.9

        assert reads[1].key == "diagnosis"
        assert reads[1].value_before == "MI"
        assert reads[1].value_after == "MI"


# ---------------------------------------------------------------------------
# LoggedMemory behavior
# ---------------------------------------------------------------------------

class TestLoggedMemory:
    def test_dict_semantics_preserved(self, mem_logger):
        mem = mem_logger.for_agent("a")
        mem["k"] = 1
        assert mem["k"] == 1
        assert "k" in mem
        assert len(mem) == 1

    def test_missing_key_raises_and_logs_nothing(self, mem_logger, store):
        mem = mem_logger.for_agent("a")
        with pytest.raises(KeyError):
            _ = mem["nonexistent"]
        diffs = store.get_full_record(TASK_ID).xai_data.memory_diffs
        assert diffs == []

    def test_complex_values_round_trip(self, mem_logger, store):
        mem = mem_logger.for_agent("a")
        mem["nested"] = {"conditions": ["MI", "PE"], "score": 0.87}
        diff = store.get_full_record(TASK_ID).xai_data.memory_diffs[0]
        assert diff.value_before is None
        assert diff.value_after == {"conditions": ["MI", "PE"], "score": 0.87}


# ---------------------------------------------------------------------------
# MemoryLogger aggregation
# ---------------------------------------------------------------------------

class TestMemoryLogger:
    def test_for_agent_returns_same_instance(self, mem_logger):
        mem1 = mem_logger.for_agent("a")
        mem2 = mem_logger.for_agent("a")
        assert mem1 is mem2

    def test_separate_memories_per_agent(self, mem_logger, store):
        mem_a = mem_logger.for_agent("specialist_a")
        mem_b = mem_logger.for_agent("specialist_b")

        mem_a["k"] = "value_a"
        mem_b["k"] = "value_b"

        assert mem_a["k"] == "value_a"
        assert mem_b["k"] == "value_b"

        diffs = store.get_full_record(TASK_ID).xai_data.memory_diffs
        write_diffs = [d for d in diffs if d.operation == "write"]
        assert len(write_diffs) == 2

        by_agent = {d.agent_id: d for d in write_diffs}
        assert by_agent["specialist_a"].value_after == "value_a"
        assert by_agent["specialist_b"].value_after == "value_b"


# ---------------------------------------------------------------------------
# ContextVar for triggered_by_event_id
# ---------------------------------------------------------------------------

class TestTriggeredByEventId:
    def test_empty_when_unset(self, mem_logger, store):
        mem = mem_logger.for_agent("a")
        mem["k"] = 1
        diff = store.get_full_record(TASK_ID).xai_data.memory_diffs[0]
        assert diff.triggered_by_event_id == ""

    def test_picks_up_contextvar(self, mem_logger, store):
        mem = mem_logger.for_agent("a")

        token = current_event_id.set("evt-123")
        try:
            mem["k"] = "v"
        finally:
            current_event_id.reset(token)

        diff = store.get_full_record(TASK_ID).xai_data.memory_diffs[0]
        assert diff.triggered_by_event_id == "evt-123"

    def test_contextvar_changes_between_operations(self, mem_logger, store):
        mem = mem_logger.for_agent("a")

        token = current_event_id.set("evt-A")
        try:
            mem["k"] = 1
        finally:
            current_event_id.reset(token)

        token = current_event_id.set("evt-B")
        try:
            _ = mem["k"]
        finally:
            current_event_id.reset(token)

        diffs = store.get_full_record(TASK_ID).xai_data.memory_diffs
        assert diffs[0].operation == "write"
        assert diffs[0].triggered_by_event_id == "evt-A"
        assert diffs[1].operation == "read"
        assert diffs[1].triggered_by_event_id == "evt-B"


# ---------------------------------------------------------------------------
# Direct LoggedMemory construction (no MemoryLogger)
# ---------------------------------------------------------------------------

class TestDirectConstruction:
    def test_initial_dict_does_not_log(self, store):
        mem = LoggedMemory(
            agent_id="a",
            store=store,
            task_id=TASK_ID,
            initial={"preload": 42},
        )
        assert mem["preload"] == 42  # this read DOES log
        diffs = store.get_full_record(TASK_ID).xai_data.memory_diffs
        # Only the read from this test, nothing from construction
        assert len(diffs) == 1
        assert diffs[0].operation == "read"
        assert diffs[0].value_after == 42
