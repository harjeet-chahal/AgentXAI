"""
Tests for the dashboard's stale-accountability detection and the
root_cause_reason surface preference.

Covers the four hypotheses from the bug report:
  1. Pre-selector records (empty `root_cause_reason`) are flagged stale.
  2. Aggregator-fallback records (selector ran but had to fall back) are
     also flagged stale.
  3. Fresh records — non-aggregator root cause + non-empty reason — are
     NOT flagged.
  4. The "What actually mattered" paragraph prefers the verbatim
     `root_cause_reason` over the bare `<action> from <agent>` synthesis.

Plus a render-safety test for the staleness banner helper.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from agentxai.ui.dashboard import (
    _FALLBACK_MARKER_SUBSTR,
    _build_what_mattered_paragraph,
    _is_aggregator_event,
    _is_stale_accountability_report,
    _render_staleness_banner,
    _staleness_message,
)


# ---------------------------------------------------------------------------
# _is_aggregator_event — sanity (mirrors the canonical check)
# ---------------------------------------------------------------------------

class TestIsAggregatorEvent:
    def test_synthesizer_read_specialist_memories_is_aggregator(self):
        assert _is_aggregator_event({
            "action": "read_specialist_memories",
            "event_type": "action",
            "agent_id": "synthesizer",
        })

    def test_orchestrator_route_is_aggregator(self):
        assert _is_aggregator_event({
            "action": "route_to_specialist_a",
            "event_type": "action",
        })
        assert _is_aggregator_event({
            "action": "handoff_to_synthesizer",
            "event_type": "action",
        })

    def test_real_decision_is_not_aggregator(self):
        assert not _is_aggregator_event({
            "action": "textbook_search",
            "event_type": "tool_call",
        })
        assert not _is_aggregator_event({
            "action": "summarize_findings",
            "event_type": "action",
        })

    def test_handles_none_and_non_dict(self):
        assert not _is_aggregator_event(None)
        assert not _is_aggregator_event("not-a-dict")
        assert not _is_aggregator_event({})


# ---------------------------------------------------------------------------
# _is_stale_accountability_report
# ---------------------------------------------------------------------------

class TestIsStaleAccountabilityReport:
    def test_empty_report_is_not_stale(self):
        # No report at all → there's nothing to be stale about.
        assert _is_stale_accountability_report(None) is False
        assert _is_stale_accountability_report({}) is False

    def test_pre_selector_record_is_stale(self):
        # Stored before root_cause_reason existed → empty string.
        report = {
            "root_cause_event_id": "evt-1",
            "root_cause_reason": "",
            "agent_responsibility_scores": {"specialist_a": 0.5,
                                              "specialist_b": 0.5},
        }
        assert _is_stale_accountability_report(report) is True

    def test_aggregator_fallback_record_is_stale(self):
        # Selector ran (root_cause_reason populated) but the trajectory
        # had no non-aggregator ancestor of the terminal — selector
        # appended the fallback marker.
        report = {
            "root_cause_event_id": "evt-1",
            "root_cause_reason": (
                "read_specialist_memories from synthesizer carried "
                "highest causal weight to the final outcome "
                "(no non-aggregator ancestor; selected from full ancestor set)"
            ),
        }
        assert _is_stale_accountability_report(report) is True
        # The marker substring matches case-insensitively too.
        report["root_cause_reason"] = report["root_cause_reason"].upper()
        assert _is_stale_accountability_report(report) is True

    def test_clean_record_is_not_stale(self):
        # Selector ran AND found a real upstream cause.
        report = {
            "root_cause_event_id": "evt-textbook",
            "root_cause_reason":
                "textbook_search from specialist_b: high-impact tool (0.85)",
        }
        assert _is_stale_accountability_report(report) is False

    def test_whitespace_only_reason_is_stale(self):
        # Record where reason was set but to whitespace — treat as stale.
        report = {"root_cause_reason": "   "}
        assert _is_stale_accountability_report(report) is True


# ---------------------------------------------------------------------------
# _staleness_message
# ---------------------------------------------------------------------------

class TestStalenessMessage:
    def test_empty_for_clean_report(self):
        report = {
            "root_cause_reason":
                "textbook_search from specialist_b: high-impact tool (0.85)",
        }
        assert _staleness_message(report) == ""

    def test_empty_for_no_report(self):
        assert _staleness_message(None) == ""
        assert _staleness_message({}) == ""

    def test_pre_selector_message(self):
        msg = _staleness_message({"root_cause_reason": ""})
        # Exact phrasing the user-facing spec asked for.
        assert "older accountability method" in msg.lower()
        assert "re-run" in msg.lower()

    def test_aggregator_fallback_message_explains_why(self):
        report = {
            "root_cause_reason": (
                "read_specialist_memories from synthesizer carried "
                "highest causal weight to the final outcome "
                "(no non-aggregator ancestor; selected from full ancestor set)"
            ),
        }
        msg = _staleness_message(report)
        # Different wording than the pre-selector case — the user needs
        # to know it's a degenerate-trajectory fallback, not a missing field.
        assert "fall back" in msg.lower() or "fell back" in msg.lower()
        assert "re-run" in msg.lower()


# ---------------------------------------------------------------------------
# _build_what_mattered_paragraph — prefers root_cause_reason verbatim
# ---------------------------------------------------------------------------

def _record_with_report(
    *,
    root_cause_reason: str = "",
    root_event: Dict[str, Any] = None,
    scores: Dict[str, float] = None,
) -> Dict[str, Any]:
    """Minimal record dict for paragraph-rendering tests."""
    trajectory: List[Dict[str, Any]] = []
    if root_event:
        trajectory.append(root_event)
    return {
        "xai_data": {
            "trajectory":   trajectory,
            "tool_calls":   [],
            "messages":     [],
            "memory_diffs": [],
            "accountability_report": {
                "agent_responsibility_scores": scores or {"specialist_b": 1.0},
                "root_cause_event_id":
                    root_event["event_id"] if root_event else "",
                "root_cause_reason": root_cause_reason,
            },
        },
    }


class TestWhatMatteredPrefersRootCauseReason:
    def test_uses_root_cause_reason_verbatim_when_present(self):
        record = _record_with_report(
            root_cause_reason=(
                "textbook_search from specialist_b: high-impact tool (0.85)"
            ),
            root_event={
                "event_id": "e1",
                "agent_id": "specialist_b",
                "event_type": "tool_call",
                "action": "textbook_search",
            },
        )
        out = _build_what_mattered_paragraph(record)
        assert "textbook_search from specialist_b: high-impact tool" in out
        # Surfaced verbatim — no naked `event_type from agent` synthesis.
        assert "tool_call from specialist_b" not in out

    def test_surfaces_fallback_marker_so_user_sees_degenerate_run(self):
        # The user-reported HIV-style scenario: aggregator root cause
        # with the fallback marker. The paragraph must surface the
        # marker so reviewers see they should re-run the task.
        record = _record_with_report(
            root_cause_reason=(
                "read_specialist_memories from synthesizer carried "
                "highest causal weight to the final outcome "
                "(no non-aggregator ancestor; selected from full ancestor set)"
            ),
            root_event={
                "event_id": "e1",
                "agent_id": "synthesizer",
                "event_type": "action",
                "action": "read_specialist_memories",
            },
        )
        out = _build_what_mattered_paragraph(record)
        assert _FALLBACK_MARKER_SUBSTR in out

    def test_falls_back_to_action_agent_when_reason_missing(self):
        # Pre-selector record — reason is empty. Paragraph should still
        # produce *something* useful from the trajectory event.
        record = _record_with_report(
            root_cause_reason="",
            root_event={
                "event_id": "e1",
                "agent_id": "specialist_b",
                "event_type": "tool_call",
                "action": "textbook_search",
            },
        )
        out = _build_what_mattered_paragraph(record)
        assert "textbook_search from specialist_b" in out

    def test_no_root_cause_at_all_skips_root_sentence(self):
        record = _record_with_report(root_cause_reason="", root_event=None)
        # Build a record where the report has scores but no root cause —
        # paragraph should still produce the responsibility sentence
        # without crashing on the missing root.
        record["xai_data"]["accountability_report"]["root_cause_event_id"] = ""
        out = _build_what_mattered_paragraph(record)
        assert "specialist_b" in out
        # No "rooted in" / "root-cause event" prose because there isn't one.
        assert "root-cause" not in out.lower()
        assert "rooted" not in out.lower()


# ---------------------------------------------------------------------------
# Banner render helper — render-safe under monkey-patched Streamlit
# ---------------------------------------------------------------------------

class TestRenderStalenessBanner:
    def test_renders_for_stale_report(self, monkeypatch):
        from agentxai.ui import dashboard
        captured: List[str] = []
        monkeypatch.setattr(
            dashboard.st, "markdown",
            lambda html, **kw: captured.append(html),
        )
        _render_staleness_banner({"root_cause_reason": ""})
        assert captured, "banner should render for stale report"
        # User-friendly phrasing from the spec.
        assert "re-run" in captured[0].lower()

    def test_silent_for_clean_report(self, monkeypatch):
        from agentxai.ui import dashboard
        captured: List[str] = []
        monkeypatch.setattr(
            dashboard.st, "markdown",
            lambda html, **kw: captured.append(html),
        )
        _render_staleness_banner({
            "root_cause_reason":
                "textbook_search from specialist_b: high-impact tool (0.85)",
        })
        assert captured == [], "banner must not render for clean report"

    def test_silent_for_missing_report(self, monkeypatch):
        from agentxai.ui import dashboard
        captured: List[str] = []
        monkeypatch.setattr(
            dashboard.st, "markdown",
            lambda html, **kw: captured.append(html),
        )
        _render_staleness_banner(None)
        _render_staleness_banner({})
        assert captured == []
