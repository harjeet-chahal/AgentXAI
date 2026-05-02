"""
Pillar 5 — Inter-Agent Communication.

MessageLogger records directed messages between agents. Each `send()` writes
an AgentMessage with acted_upon=False; the counterfactual engine later calls
`mark_acted_upon(message_id, description)` when it determines that the
message actually changed the receiver's behavior.

`build_communication_graph(store, task_id)` returns a NetworkX
``MultiDiGraph`` with agents as nodes and one edge per message,
carrying message_type / acted_upon / content as edge attributes.
A multi-graph is needed because two agents may exchange more than one
message and each one is a distinct edge for accountability purposes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import networkx as nx

from agentxai.data.schemas import AgentMessage
from agentxai.store.trajectory_store import TrajectoryStore


class MessageLogger:
    """Record inter-agent messages for one task."""

    def __init__(self, store: TrajectoryStore, task_id: str) -> None:
        self.store = store
        self.task_id = task_id
        self._messages: Dict[str, AgentMessage] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(
        self,
        sender: str,
        receiver: str,
        message_type: str,
        content: Dict[str, Any],
    ) -> AgentMessage:
        """Record a new message with acted_upon=False."""
        message = AgentMessage(
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            content=dict(content or {}),
            acted_upon=False,
            behavior_change_description="",
        )
        self._messages[message.message_id] = message
        self.store.save_message(self.task_id, message)
        return message

    def mark_acted_upon(self, message_id: str, description: str) -> AgentMessage:
        """Flip acted_upon→True and attach a human-readable description."""
        message = self._messages.get(message_id) or self._load_from_store(message_id)
        if message is None:
            raise KeyError(
                f"Message {message_id!r} not found for task {self.task_id!r}."
            )

        message.acted_upon = True
        message.behavior_change_description = str(description)
        self._messages[message_id] = message
        self.store.save_message(self.task_id, message)
        return message

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_from_store(self, message_id: str) -> Optional[AgentMessage]:
        try:
            record = self.store.get_full_record(self.task_id)
        except KeyError:
            return None
        for m in record.xai_data.messages:
            if m.message_id == message_id:
                return m
        return None


# ---------------------------------------------------------------------------
# Communication-graph helper
# ---------------------------------------------------------------------------

def build_communication_graph(
    store: TrajectoryStore,
    task_id: str,
) -> nx.MultiDiGraph:
    """
    Build a directed multigraph of all messages for one task.

    Nodes are agent_ids. Each message is one directed edge from sender
    to receiver with attrs: message_id, message_type, acted_upon,
    content, behavior_change_description, timestamp. Multiple messages
    between the same agent pair produce multiple edges (hence the
    MultiDiGraph). Missing tasks return an empty graph.

    Note: `nx.MultiDiGraph` IS-A `nx.DiGraph`, so callers still expecting
    a DiGraph keep working — but the return type now states the truth.
    """
    graph: nx.MultiDiGraph = nx.MultiDiGraph()

    try:
        record = store.get_full_record(task_id)
    except KeyError:
        return graph

    for msg in record.xai_data.messages:
        if msg.sender:
            graph.add_node(msg.sender)
        if msg.receiver:
            graph.add_node(msg.receiver)
        if msg.sender and msg.receiver:
            graph.add_edge(
                msg.sender,
                msg.receiver,
                key=msg.message_id,
                message_id=msg.message_id,
                message_type=msg.message_type,
                acted_upon=msg.acted_upon,
                content=msg.content,
                behavior_change_description=msg.behavior_change_description,
                timestamp=msg.timestamp,
            )

    return graph
