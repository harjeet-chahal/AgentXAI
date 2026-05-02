"""
Tests for agentxai/xai/message_logger.py — Pillar 5.
"""

from __future__ import annotations

import networkx as nx
import pytest

from agentxai.data.schemas import AgentXAIRecord
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.message_logger import MessageLogger, build_communication_graph


TASK_ID = "MSG-TEST-001"


@pytest.fixture()
def store() -> TrajectoryStore:
    s = TrajectoryStore(db_url="sqlite:///:memory:")
    s.save_task(AgentXAIRecord(task_id=TASK_ID, source="test"))
    return s


@pytest.fixture()
def msg_logger(store: TrajectoryStore) -> MessageLogger:
    return MessageLogger(store=store, task_id=TASK_ID)


# ---------------------------------------------------------------------------
# send()
# ---------------------------------------------------------------------------

class TestSend:
    def test_writes_with_defaults(self, msg_logger, store):
        m = msg_logger.send(
            sender="specialist_a",
            receiver="orchestrator",
            message_type="finding",
            content={"top_condition": "MI"},
        )
        assert m.message_id
        assert m.sender == "specialist_a"
        assert m.receiver == "orchestrator"
        assert m.message_type == "finding"
        assert m.content == {"top_condition": "MI"}
        assert m.acted_upon is False
        assert m.behavior_change_description == ""

        persisted = store.get_full_record(TASK_ID).xai_data.messages[0]
        assert persisted.message_id == m.message_id
        assert persisted.acted_upon is False


# ---------------------------------------------------------------------------
# mark_acted_upon()
# ---------------------------------------------------------------------------

class TestMarkActedUpon:
    def test_updates_flag_and_description(self, msg_logger, store):
        m = msg_logger.send("a", "b", "finding", {"k": 1})
        msg_logger.mark_acted_upon(
            m.message_id, "Receiver switched to specialist_c after this message."
        )
        persisted = store.get_full_record(TASK_ID).xai_data.messages[0]
        assert persisted.acted_upon is True
        assert "switched" in persisted.behavior_change_description

    def test_works_without_in_memory_cache(self, store):
        first = MessageLogger(store=store, task_id=TASK_ID)
        m = first.send("a", "b", "finding", {})

        second = MessageLogger(store=store, task_id=TASK_ID)
        second.mark_acted_upon(m.message_id, "retroactive")

        persisted = store.get_full_record(TASK_ID).xai_data.messages[0]
        assert persisted.acted_upon is True
        assert persisted.behavior_change_description == "retroactive"

    def test_unknown_id_raises(self, msg_logger):
        with pytest.raises(KeyError, match="not found"):
            msg_logger.mark_acted_upon("nonexistent", "n/a")


# ---------------------------------------------------------------------------
# build_communication_graph()
# ---------------------------------------------------------------------------

class TestBuildCommunicationGraph:
    def test_three_messages_two_agents(self, msg_logger, store):
        m1 = msg_logger.send("specialist_a", "orchestrator", "finding", {"top": "MI"})
        m2 = msg_logger.send("orchestrator", "specialist_a", "routing", {"next": "lab"})
        m3 = msg_logger.send("specialist_a", "orchestrator", "finding", {"top": "PE"})

        graph = build_communication_graph(store, TASK_ID)

        # Two agents
        assert graph.number_of_nodes() == 2
        assert set(graph.nodes()) == {"specialist_a", "orchestrator"}

        # Three directed edges (one per message, MultiDiGraph preserves parallels)
        assert graph.number_of_edges() == 3
        assert graph.number_of_edges("specialist_a", "orchestrator") == 2
        assert graph.number_of_edges("orchestrator", "specialist_a") == 1

    def test_edge_attrs_preserved(self, msg_logger, store):
        m = msg_logger.send(
            "a", "b", "correction", {"note": "retry with contrast"}
        )
        graph = build_communication_graph(store, TASK_ID)

        attrs = graph.get_edge_data("a", "b", key=m.message_id)
        assert attrs is not None
        assert attrs["message_id"] == m.message_id
        assert attrs["message_type"] == "correction"
        assert attrs["acted_upon"] is False
        assert attrs["content"] == {"note": "retry with contrast"}

    def test_mark_acted_upon_reflected_in_graph(self, msg_logger, store):
        m = msg_logger.send("a", "b", "finding", {})
        msg_logger.mark_acted_upon(m.message_id, "receiver rerouted")

        graph = build_communication_graph(store, TASK_ID)
        attrs = graph.get_edge_data("a", "b", key=m.message_id)
        assert attrs["acted_upon"] is True
        assert attrs["behavior_change_description"] == "receiver rerouted"

    def test_is_a_digraph(self, store):
        graph = build_communication_graph(store, TASK_ID)
        assert isinstance(graph, nx.DiGraph)

    def test_returns_a_multidigraph(self, store):
        """
        Regression: the function used to be annotated as `-> nx.DiGraph`
        despite returning a MultiDiGraph. The annotation is now honest.
        Asserting the runtime type lets future readers spot if someone
        silently downgrades the return.
        """
        graph = build_communication_graph(store, TASK_ID)
        assert isinstance(graph, nx.MultiDiGraph)
        # Sanity: MultiDiGraph IS a DiGraph too — old callers don't break.
        assert isinstance(graph, nx.DiGraph)

    def test_multidigraph_preserves_repeat_edges(self, store):
        """Two messages between the same pair → two distinct edges."""
        from agentxai.data.schemas import AgentMessage

        store.save_message(TASK_ID, AgentMessage(
            sender="a", receiver="b", message_type="finding",
            content={"first": True},
        ))
        store.save_message(TASK_ID, AgentMessage(
            sender="a", receiver="b", message_type="finding",
            content={"second": True},
        ))
        graph = build_communication_graph(store, TASK_ID)
        # Both edges survived — a plain DiGraph would have collapsed them.
        assert graph.number_of_edges("a", "b") == 2

    def test_unknown_task_returns_empty(self, store):
        graph = build_communication_graph(store, "DOES_NOT_EXIST")
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_empty_task_returns_empty_graph(self, store):
        # Task exists but has no messages.
        graph = build_communication_graph(store, TASK_ID)
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0
