"""
Tests for ``eval/evaluate_accountability.py``.

Each metric is tested in isolation with a hand-crafted task record, plus
an end-to-end test of the aggregator that builds a tiny store with two
synthetic tasks and verifies the printed summary mentions the headline
counts.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
from sqlalchemy import text

from agentxai.data.schemas import (
    AccountabilityReport,
    AgentMessage,
    AgentXAIRecord,
    CausalEdge,
    CausalGraph,
    MemoryDiff,
    ToolUseEvent,
    TrajectoryEvent,
    XAIData,
)
from agentxai.store.trajectory_store import TrajectoryStore
from eval.evaluate_accountability import (
    DEFAULT_EMPTY_RESP_THRESHOLD,
    _aggregate_empty_agent,
    _aggregate_faithfulness,
    _aggregate_impact_alignment,
    _aggregate_root_cause_validity,
    empty_agent_penalty_for_task,
    evaluate_accountability,
    faithfulness_for_task,
    format_summary,
    impact_alignment_for_task,
    main,
    root_cause_validity_for_task,
)


# ---------------------------------------------------------------------------
# Helpers — build synthetic tasks in-memory
# ---------------------------------------------------------------------------

def _empty_record(
    *,
    task_id: str = "T",
    scores: Dict[str, float] = None,
    tool_calls: List[ToolUseEvent] = None,
    messages: List[AgentMessage] = None,
    memory_diffs: List[MemoryDiff] = None,
    trajectory: List[TrajectoryEvent] = None,
    most_impactful_tool_call_id: str = "",
    most_influential_message_id: str = "",
    root_cause_event_id: str = "",
    root_cause_reason: str = "",
    causal_chain: List[str] = None,
    memory_usage: List = None,
) -> AgentXAIRecord:
    """Build a hydrated AgentXAIRecord without touching the store."""
    report = AccountabilityReport(
        task_id=task_id,
        agent_responsibility_scores=dict(scores or {}),
        most_impactful_tool_call_id=most_impactful_tool_call_id,
        most_influential_message_id=most_influential_message_id,
        root_cause_event_id=root_cause_event_id,
        root_cause_reason=root_cause_reason,
        causal_chain=list(causal_chain or []),
        memory_usage=list(memory_usage or []),
    )
    xai = XAIData(
        trajectory=list(trajectory or []),
        tool_calls=list(tool_calls or []),
        messages=list(messages or []),
        memory_diffs=list(memory_diffs or []),
        causal_graph=CausalGraph(),
        accountability_report=report,
    )
    return AgentXAIRecord(
        task_id=task_id, source="test",
        input={}, ground_truth={}, system_output={},
        xai_data=xai,
    )


# ---------------------------------------------------------------------------
# Metric 1 — empty-agent penalty
# ---------------------------------------------------------------------------

class TestEmptyAgentPenalty:
    def test_flags_empty_agent_with_high_responsibility(self):
        # Specialist A has 0.50 responsibility but no observable signals.
        rec = _empty_record(
            task_id="T1",
            scores={"specialist_a": 0.50, "specialist_b": 0.50},
            tool_calls=[ToolUseEvent(
                tool_name="symptom_lookup", called_by="specialist_b",
                downstream_impact_score=0.8,
            )],
        )
        out = empty_agent_penalty_for_task(rec)
        a = next(s for s in out["specialists"] if s["agent_id"] == "specialist_a")
        assert a["is_empty"] is True
        assert a["is_undeserved"] is True
        b = next(s for s in out["specialists"] if s["agent_id"] == "specialist_b")
        assert b["is_empty"] is False
        assert b["is_undeserved"] is False

    def test_low_responsibility_empty_agent_not_flagged_undeserved(self):
        rec = _empty_record(
            task_id="T2",
            scores={"specialist_a": 0.05, "specialist_b": 0.95},
        )
        out = empty_agent_penalty_for_task(rec)
        a = next(s for s in out["specialists"] if s["agent_id"] == "specialist_a")
        assert a["is_empty"] is True
        # Below threshold (default 0.20) → not undeserved.
        assert a["is_undeserved"] is False

    def test_aggregator_counts_undeserved_correctly(self):
        per_task = [
            empty_agent_penalty_for_task(_empty_record(
                task_id="T1",
                scores={"specialist_a": 0.50, "specialist_b": 0.50},
                tool_calls=[ToolUseEvent(called_by="specialist_b",
                                          downstream_impact_score=0.5)],
            )),
            empty_agent_penalty_for_task(_empty_record(
                task_id="T2",
                scores={"specialist_a": 0.10, "specialist_b": 0.90},
            )),
        ]
        agg = _aggregate_empty_agent(per_task)
        assert agg["n_tasks"] == 2
        # Empty + undeserved (responsibility ≥ 0.20):
        #   T1.specialist_a (empty, 0.50)  → undeserved
        #   T1.specialist_b has a tool      → NOT empty
        #   T2.specialist_a (empty, 0.10)  → empty but below threshold
        #   T2.specialist_b (empty, 0.90)  → undeserved
        # → 2 undeserved across both tasks.
        assert agg["n_undeserved_agents"] == 2
        assert agg["mean_resp_of_empty_agents"] > 0
        # Empty responsibilities: 0.50 + 0.10 + 0.90.
        assert agg["max_resp_of_empty_agents"] == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# Metric 2 — impact alignment
# ---------------------------------------------------------------------------

class TestImpactAlignment:
    def test_aligned_when_top_agent_owns_top_tool(self):
        rec = _empty_record(
            task_id="T1",
            scores={"specialist_b": 0.85, "specialist_a": 0.15},
            tool_calls=[
                ToolUseEvent(tool_name="pubmed_search",
                             called_by="specialist_b",
                             downstream_impact_score=0.9),
                ToolUseEvent(tool_name="symptom_lookup",
                             called_by="specialist_a",
                             downstream_impact_score=0.2),
            ],
        )
        out = impact_alignment_for_task(rec)
        assert out["aligned_with_tool"] is True
        assert out["aligned_either"] is True
        assert out["skip"] is False

    def test_aligned_when_top_agent_sent_top_message(self):
        rec = _empty_record(
            task_id="T1",
            scores={"specialist_a": 0.7, "specialist_b": 0.3},
            messages=[
                AgentMessage(message_id="m1", sender="specialist_a",
                             receiver="synthesizer", acted_upon=True),
            ],
            most_influential_message_id="m1",
        )
        out = impact_alignment_for_task(rec)
        assert out["aligned_with_message"] is True
        assert out["aligned_either"] is True

    def test_misaligned_when_top_agent_neither(self):
        # Top responsible is A but B owns the impactful tool AND sent
        # the influential message → misalignment.
        rec = _empty_record(
            task_id="T1",
            scores={"specialist_a": 0.7, "specialist_b": 0.3},
            tool_calls=[ToolUseEvent(called_by="specialist_b",
                                      downstream_impact_score=0.9)],
            messages=[AgentMessage(message_id="m1", sender="specialist_b",
                                    acted_upon=True)],
            most_influential_message_id="m1",
        )
        out = impact_alignment_for_task(rec)
        assert out["aligned_with_tool"] is False
        assert out["aligned_with_message"] is False
        assert out["aligned_either"] is False

    def test_skip_when_no_signals_to_align(self):
        # No tools, no message → nothing to align against.
        rec = _empty_record(
            task_id="T1",
            scores={"specialist_a": 0.5, "specialist_b": 0.5},
        )
        out = impact_alignment_for_task(rec)
        assert out["skip"] is True

    def test_zero_impact_tool_excluded(self):
        # Tool exists but impact_score=0 → as if no tool. Skipped.
        rec = _empty_record(
            task_id="T1",
            scores={"specialist_a": 0.5, "specialist_b": 0.5},
            tool_calls=[ToolUseEvent(called_by="specialist_b",
                                      downstream_impact_score=0.0)],
        )
        out = impact_alignment_for_task(rec)
        assert out["highest_impact_tool_owner"] is None
        assert out["skip"] is True

    def test_aggregator_excludes_skipped(self):
        per_task = [
            impact_alignment_for_task(_empty_record(
                task_id="T1",
                scores={"specialist_b": 0.85},
                tool_calls=[ToolUseEvent(called_by="specialist_b",
                                          downstream_impact_score=0.9)],
            )),
            # Skipped — no tool / no msg.
            impact_alignment_for_task(_empty_record(
                task_id="T2",
                scores={"specialist_a": 0.5, "specialist_b": 0.5},
            )),
        ]
        agg = _aggregate_impact_alignment(per_task)
        assert agg["n_tasks"] == 2
        assert agg["n_tasks_evaluated"] == 1
        assert agg["n_tasks_skipped"] == 1
        assert agg["alignment_rate_either"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Metric 3 — root-cause validity
# ---------------------------------------------------------------------------

class TestRootCauseValidity:
    def test_valid_when_in_chain_and_non_aggregator(self):
        rec = _empty_record(
            task_id="T1",
            scores={"specialist_b": 0.9},
            trajectory=[TrajectoryEvent(
                event_id="e1", agent_id="specialist_b",
                event_type="tool_call", action="pubmed_search",
            )],
            root_cause_event_id="e1",
            causal_chain=["e1", "e_term"],
        )
        out = root_cause_validity_for_task(rec)
        assert out["in_chain"] is True
        assert out["is_aggregator"] is False
        assert out["is_valid"] is True

    def test_invalid_when_aggregator_without_fallback_marker(self):
        rec = _empty_record(
            task_id="T1",
            trajectory=[TrajectoryEvent(
                event_id="e1", agent_id="synthesizer",
                event_type="agent_action", action="read_specialist_memories",
            )],
            root_cause_event_id="e1",
            causal_chain=["e1"],
            root_cause_reason="",   # no fallback marker
        )
        out = root_cause_validity_for_task(rec)
        assert out["is_aggregator"] is True
        assert out["fallback_marker"] is False
        assert out["is_valid"] is False

    def test_valid_when_aggregator_with_fallback_marker(self):
        rec = _empty_record(
            task_id="T1",
            trajectory=[TrajectoryEvent(
                event_id="e1", agent_id="synthesizer",
                event_type="agent_action", action="read_specialist_memories",
            )],
            root_cause_event_id="e1",
            causal_chain=["e1"],
            root_cause_reason=(
                "read_specialist_memories from synthesizer "
                "(no non-aggregator ancestor; selected from full ancestor set)"
            ),
        )
        out = root_cause_validity_for_task(rec)
        assert out["is_aggregator"] is True
        assert out["fallback_marker"] is True
        assert out["is_valid"] is True

    def test_invalid_when_root_not_in_chain(self):
        rec = _empty_record(
            task_id="T1",
            trajectory=[TrajectoryEvent(
                event_id="e1", agent_id="specialist_b",
                event_type="tool_call", action="pubmed_search",
            )],
            root_cause_event_id="e1",
            causal_chain=["other_event", "e_term"],   # e1 missing
        )
        out = root_cause_validity_for_task(rec)
        assert out["in_chain"] is False
        assert out["is_valid"] is False

    def test_skip_when_no_root_cause(self):
        rec = _empty_record(task_id="T1", root_cause_event_id="")
        out = root_cause_validity_for_task(rec)
        assert out["skip"] is True

    def test_aggregator_counts_validity(self):
        per_task = [
            root_cause_validity_for_task(_empty_record(
                task_id="T1",
                trajectory=[TrajectoryEvent(event_id="e1", action="pubmed_search",
                                             event_type="tool_call")],
                root_cause_event_id="e1", causal_chain=["e1"],
            )),
            root_cause_validity_for_task(_empty_record(
                task_id="T2",
                trajectory=[TrajectoryEvent(event_id="e1",
                                             action="read_specialist_memories",
                                             event_type="agent_action")],
                root_cause_event_id="e1", causal_chain=["e1"],
            )),
        ]
        agg = _aggregate_root_cause_validity(per_task)
        assert agg["n_tasks"] == 2
        assert agg["n_tasks_evaluated"] == 2
        assert agg["n_valid"] == 1
        assert agg["validity_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Metric 4 — faithfulness (reads counterfactual_runs)
# ---------------------------------------------------------------------------

def _seed_cf_runs(store: TrajectoryStore, task_id: str,
                  deltas: Dict[str, float]) -> None:
    """Seed agent_output rows in counterfactual_runs."""
    # Ensure the table exists (created on first CounterfactualEngine init).
    from agentxai.xai.counterfactual_engine import CounterfactualEngine

    class _StubPipeline:
        def resume_from(self, snapshot, overrides):
            return {"final_diagnosis": "X", "confidence": 0.5}

    CounterfactualEngine(store=store, pipeline=_StubPipeline(), task_id=task_id)
    with store._engine.connect() as conn:
        for i, (agent, delta) in enumerate(deltas.items()):
            conn.execute(
                text(
                    "INSERT INTO counterfactual_runs "
                    "(run_id, task_id, perturbation_type, target_id, "
                    " baseline_value_json, original_outcome_json, "
                    " perturbed_outcome_json, outcome_delta) "
                    "VALUES (:r, :t, 'agent_output', :tid, '{}', '{}', '{}', :d)"
                ),
                {"r": f"run-{task_id}-{i}", "t": task_id, "tid": agent, "d": delta},
            )
        conn.commit()


class TestFaithfulness:
    def test_top_agent_higher_delta_is_faithful(self):
        store = TrajectoryStore(db_url="sqlite:///:memory:")
        store.save_task(AgentXAIRecord(task_id="T1", source="test"))
        rec = _empty_record(
            task_id="T1",
            scores={"specialist_b": 0.85, "specialist_a": 0.15},
        )
        _seed_cf_runs(store, "T1",
                      {"specialist_b": 0.9, "specialist_a": 0.2})
        out = faithfulness_for_task(rec, store)
        assert out["skip"] is False
        assert out["top_agent"] == "specialist_b"
        assert out["delta_gap"] == pytest.approx(0.7)
        assert out["is_faithful"] is True

    def test_low_agent_higher_delta_is_unfaithful(self):
        store = TrajectoryStore(db_url="sqlite:///:memory:")
        store.save_task(AgentXAIRecord(task_id="T1", source="test"))
        rec = _empty_record(
            task_id="T1",
            scores={"specialist_b": 0.85, "specialist_a": 0.15},
        )
        # Bug-ish report: specialist_a was scored low but actually
        # produced a bigger outcome change when zeroed.
        _seed_cf_runs(store, "T1",
                      {"specialist_b": 0.2, "specialist_a": 0.9})
        out = faithfulness_for_task(rec, store)
        assert out["delta_gap"] < 0
        assert out["is_faithful"] is False

    def test_skip_when_no_cf_data(self):
        store = TrajectoryStore(db_url="sqlite:///:memory:")
        store.save_task(AgentXAIRecord(task_id="T1", source="test"))
        rec = _empty_record(
            task_id="T1",
            scores={"specialist_b": 0.85, "specialist_a": 0.15},
        )
        out = faithfulness_for_task(rec, store)
        assert out["skip"] is True

    def test_skip_when_only_one_specialist(self):
        store = TrajectoryStore(db_url="sqlite:///:memory:")
        store.save_task(AgentXAIRecord(task_id="T1", source="test"))
        rec = _empty_record(
            task_id="T1",
            scores={"specialist_a": 1.0},
        )
        out = faithfulness_for_task(rec, store)
        assert out["skip"] is True

    def test_aggregator_handles_mix_of_skip_and_scored(self):
        per_task = [
            {"skip": False, "is_faithful": True,  "delta_gap": 0.5},
            {"skip": False, "is_faithful": False, "delta_gap": -0.2},
            {"skip": True},
        ]
        agg = _aggregate_faithfulness(per_task)
        assert agg["n_tasks_with_data"] == 2
        assert agg["n_faithful"] == 1
        assert agg["alignment_rate"] == pytest.approx(0.5)
        assert agg["mean_delta_gap"] == pytest.approx(0.15)

    def test_aggregator_zero_data(self):
        agg = _aggregate_faithfulness([{"skip": True}, {"skip": True}])
        assert agg["n_tasks_with_data"] == 0
        assert agg["mean_delta_gap"] == 0.0


# ---------------------------------------------------------------------------
# End-to-end driver
# ---------------------------------------------------------------------------

def _seed_full_store(store: TrajectoryStore) -> List[str]:
    """
    Two synthetic tasks:
      T-good — top agent owns the impactful tool, root cause is real,
               cf delta ordered correctly. All metrics pass.
      T-bad  — empty Specialist A scored 0.5 (undeserved), no impactful
               tool to align against, root cause is an aggregator,
               cf delta is inverted (low agent has bigger delta).
    """
    ids = []

    # T-good
    s_good = AgentXAIRecord(
        task_id="T-good", source="test",
        input={}, ground_truth={},
        system_output={"final_diagnosis": "MI", "confidence": 0.9, "correct": True},
    )
    store.save_task(s_good)
    e1 = TrajectoryEvent(event_id="g_e1", agent_id="specialist_b",
                         event_type="tool_call", action="pubmed_search",
                         timestamp=1.0)
    store.save_event("T-good", e1)
    tool = ToolUseEvent(tool_call_id="g_t1", tool_name="pubmed_search",
                        called_by="specialist_b", timestamp=1.0,
                        downstream_impact_score=0.9)
    store.save_tool_call("T-good", tool)
    msg = AgentMessage(message_id="g_m1", sender="specialist_b",
                       receiver="synthesizer", acted_upon=True)
    store.save_message("T-good", msg)
    store.save_memory_diff("T-good", MemoryDiff(
        agent_id="specialist_b", operation="write",
        key="top_evidence", value_after=["MI"],
    ))
    store.save_accountability_report(AccountabilityReport(
        task_id="T-good",
        final_outcome="MI", outcome_correct=True,
        agent_responsibility_scores={"specialist_b": 0.85, "specialist_a": 0.15},
        most_impactful_tool_call_id="g_t1",
        most_influential_message_id="g_m1",
        root_cause_event_id="g_e1",
        causal_chain=["g_e1"],
        root_cause_reason="pubmed_search from specialist_b: high-impact tool (0.9)",
    ))
    _seed_cf_runs(store, "T-good", {"specialist_b": 0.9, "specialist_a": 0.2})
    ids.append("T-good")

    # T-bad
    s_bad = AgentXAIRecord(
        task_id="T-bad", source="test",
        input={}, ground_truth={},
        system_output={"final_diagnosis": "?", "confidence": 0.5, "correct": False},
    )
    store.save_task(s_bad)
    e2 = TrajectoryEvent(event_id="b_e1", agent_id="synthesizer",
                         event_type="agent_action",
                         action="read_specialist_memories",
                         timestamp=2.0)
    store.save_event("T-bad", e2)
    # No tools, no acted-upon messages.
    store.save_accountability_report(AccountabilityReport(
        task_id="T-bad",
        final_outcome="?", outcome_correct=False,
        agent_responsibility_scores={"specialist_a": 0.5, "specialist_b": 0.5},
        # Aggregator root cause without fallback marker → invalid.
        root_cause_event_id="b_e1",
        causal_chain=["b_e1"],
        root_cause_reason="read_specialist_memories from synthesizer",
    ))
    _seed_cf_runs(store, "T-bad", {"specialist_a": 0.1, "specialist_b": 0.7})
    ids.append("T-bad")

    return ids


class TestEvaluateAccountabilityEndToEnd:
    def test_aggregates_metrics_over_two_tasks(self):
        store = TrajectoryStore(db_url="sqlite:///:memory:")
        _seed_full_store(store)

        results = evaluate_accountability(store, limit=10)
        m = results["metrics"]

        # Metric 1: T-bad has Specialist A at 0.5 with empty signals →
        # 1 undeserved.
        assert m["empty_agent_penalty"]["n_undeserved_agents"] >= 1
        assert m["empty_agent_penalty"]["n_tasks_with_empty_agent"] >= 1

        # Metric 2: T-good aligns (specialist_b is top, owns the tool,
        # sent the influential message). T-bad is skipped (no tool / msg).
        assert m["impact_alignment"]["n_aligned_either"] == 1
        assert m["impact_alignment"]["alignment_rate_either"] == pytest.approx(1.0)

        # Metric 3: T-good's root cause is a real tool event → valid.
        # T-bad's root cause is read_specialist_memories with no fallback
        # marker → invalid.
        assert m["root_cause_validity"]["n_valid"] == 1
        assert m["root_cause_validity"]["validity_rate"] == pytest.approx(0.5)

        # Metric 4: T-good is faithful (top=specialist_b cf delta 0.9 >
        # specialist_a 0.2). T-bad is unfaithful (top=specialist_a or
        # specialist_b with a tied 0.5 score; max picks first by
        # iteration order = 'specialist_a'; delta_gap = 0.1 - 0.7 < 0).
        f = m["faithfulness"]
        assert f["n_tasks_with_data"] == 2
        assert f["n_faithful"] == 1
        assert f["mean_delta_gap"] == pytest.approx((0.7 + (-0.6)) / 2, abs=0.01)

    def test_summary_is_human_readable(self):
        store = TrajectoryStore(db_url="sqlite:///:memory:")
        _seed_full_store(store)
        results = evaluate_accountability(store, limit=10)
        text_out = format_summary(results)

        # Spot-check section headers and key labels show up.
        for label in (
            "Empty-agent penalty",
            "Impact alignment",
            "Root-cause validity",
            "Faithfulness",
            "AgentXAI accountability evaluation",
        ):
            assert label in text_out

    def test_evaluate_handles_empty_store(self):
        store = TrajectoryStore(db_url="sqlite:///:memory:")
        results = evaluate_accountability(store, limit=10)
        assert results["n_records_loaded"] == 0
        # Every aggregate is well-defined for zero tasks.
        for m in results["metrics"].values():
            assert m["n_tasks"] == 0


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

class TestCLISmoke:
    def test_main_runs_against_in_memory_db(self, tmp_path, capsys, monkeypatch):
        # Build an in-memory DB, then point the CLI at it.
        store = TrajectoryStore(db_url="sqlite:///:memory:")
        _seed_full_store(store)

        # main() builds its own TrajectoryStore from --db; we need to
        # point it at our seeded store instead. Easiest: patch
        # TrajectoryStore's constructor to return our existing instance.
        from eval import evaluate_accountability as mod
        monkeypatch.setattr(mod, "TrajectoryStore", lambda *a, **kw: store)

        out_path = tmp_path / "results.json"
        rc = main(["--limit", "10", "--out-json", str(out_path)])
        assert rc == 0

        captured = capsys.readouterr().out
        assert "AgentXAI accountability evaluation" in captured

        # JSON file written and parses.
        with open(out_path) as fh:
            data = json.load(fh)
        assert "metrics" in data
        assert set(data["metrics"]) == {
            "empty_agent_penalty",
            "impact_alignment",
            "root_cause_validity",
            "faithfulness",
        }
