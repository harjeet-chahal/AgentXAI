"""
Specialist B — Evidence Retriever.

Pipeline (one ``active_plan``):

1. ``extract_candidate_conditions`` — LLM produces a short list of plausible
                                       conditions to look up
2. ``textbook_search``              — top-k passages for the case from a
                                       local FAISS index over 18 medical
                                       textbooks (Harrison, Robbins, First
                                       Aid, etc.). Whether the call runs
                                       and what query it uses is decided
                                       per-case by the LLM.
3. ``guideline_lookup``             — fuzzy guideline match. Whether the
                                       call runs is also LLM-decided; the
                                       LLM may supply a single override
                                       query or fall through to looking up
                                       each candidate condition.
4. ``summarize_findings``           — write four memory keys + send a
                                       ``finding`` message to the Orchestrator

Both retrieval tools are LLM-gated so that Pillar 3 (Tool Provenance) varies
case-by-case rather than recording the same fixed call sequence every run.
The LLM's reasoning is recorded as the trajectory event's ``outcome``
field so it surfaces in the dashboard.

Memory keys written:

    retrieved_docs        : List[dict]   # full textbook_search response
    top_evidence          : List[dict]   # top-N (default 3) {doc_id, score, source_file, snippet}
    guideline_matches     : List[dict]   # one per candidate condition (match=None if missed)
    retrieval_confidence  : float        # mean score of top_evidence (clamped to [0, 1])
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from agentxai.agents._llm_utils import (
    extract_text,
    parse_json_list,
    parse_json_object,
)
from agentxai.agents.base import TracedAgent, make_default_llm
from agentxai.tools.guideline_lookup import guideline_lookup as _real_guideline_lookup
from agentxai.tools.textbook_search import textbook_search as _real_textbook_search


_CONDITION_PROMPT = (
    "You are a clinical reasoner. From the patient case below, list up to 5 "
    "plausible candidate conditions (concise diagnosis names) that should be "
    "looked up in clinical guidelines.\n"
    "Return ONLY a JSON array of strings. No prose, no markdown.\n\n"
    "Patient case:\n{case}"
)

_EVIDENCE_DECISION_PROMPT = (
    "Decide which retrieval tools to call for this case.\n"
    "Return ONLY a JSON object of the form:\n"
    "{{\n"
    '  "textbook_search":  {{"run": bool, "reason": str, "query": str}},\n'
    '  "guideline_lookup": {{"run": bool, "reason": str, "query": str}}\n'
    "}}\n"
    "Use \"query\" to override the default query. Leave \"query\" empty to use "
    "the default (the full patient case for textbook_search; the candidate "
    "conditions list for guideline_lookup).\n\n"
    "Patient case:\n{case}\n"
    "Candidate conditions: {candidates}"
)

_SNIPPET_CHARS = 240


class SpecialistB(TracedAgent):
    """Evidence retriever — textbook_search + guideline_lookup + LLM candidate gen."""

    def __init__(
        self,
        *,
        agent_id: str = "specialist_b",
        textbook_search_fn: Optional[Callable[..., List[dict]]] = None,
        guideline_lookup_fn: Optional[Callable[[str], dict]] = None,
        orchestrator_id: str = "orchestrator",
        k_docs: int = 5,
        top_evidence_n: int = 3,
        llm: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            llm=llm if llm is not None else make_default_llm(),
            **kwargs,
        )
        self.textbook_search_fn = textbook_search_fn or _real_textbook_search
        self.guideline_lookup_fn = guideline_lookup_fn or _real_guideline_lookup
        self.orchestrator_id = orchestrator_id
        self.k_docs = k_docs
        self.top_evidence_n = top_evidence_n

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self, input_payload: dict) -> dict:
        case = (
            input_payload.get("patient_case")
            or input_payload.get("question")
            or ""
        )

        available_actions: List[str] = [
            "extract_candidate_conditions",
            "textbook_search",
            "guideline_lookup",
            "summarize_findings",
        ]
        plan = self.generate_plan(case, available_actions)

        candidates: List[str] = []
        retrieved_docs: List[dict] = []
        top_evidence: List[dict] = []
        guideline_matches: List[dict] = []
        findings: Dict[str, Any] = {}
        decision: Dict[str, Dict[str, Any]] = {}

        with self.active_plan(plan):
            for action in plan:
                if action == "extract_candidate_conditions":
                    candidates = self._extract_candidates(case)
                    # Decide tool gating once we know the candidates so
                    # the LLM can incorporate them in its reasoning.
                    decision = self._decide_evidence_tools(case, candidates)
                    self.log_action(
                        "extract_candidate_conditions",
                        {"case_chars": len(case)},
                        outcome=f"{len(candidates)} candidates",
                    )
                elif action == "textbook_search":
                    tb_decision = decision.get("textbook_search") or {}
                    run_tb = bool(tb_decision.get("run", True))
                    tb_query = str(tb_decision.get("query") or "").strip() or case
                    tb_reason = str(tb_decision.get("reason") or "")
                    if run_tb:
                        retrieved_docs = list(
                            self.textbook_search_fn(tb_query, k=self.k_docs) or []
                        )
                        top_evidence = [
                            {
                                "doc_id":      d.get("doc_id", ""),
                                "score":       float(d.get("score", 0.0)),
                                "source_file": d.get("source_file", ""),
                                "snippet":     str(d.get("text", ""))[:_SNIPPET_CHARS],
                            }
                            for d in retrieved_docs[: self.top_evidence_n]
                        ]
                        outcome = (
                            f"{len(retrieved_docs)} docs, top score "
                            f"{(top_evidence[0]['score'] if top_evidence else 0.0):.3f}"
                        )
                    else:
                        outcome = "skipped by LLM"
                    if tb_reason:
                        outcome = f"{outcome} | reasoning: {tb_reason}"
                    self.log_action(
                        "textbook_search",
                        {
                            "query_chars": len(tb_query),
                            "k":           self.k_docs,
                            "ran":         run_tb,
                            "query":       tb_query if run_tb else "",
                        },
                        outcome=outcome,
                    )
                elif action == "guideline_lookup":
                    gl_decision = decision.get("guideline_lookup") or {}
                    run_gl = bool(gl_decision.get("run", True))
                    gl_query = str(gl_decision.get("query") or "").strip()
                    gl_reason = str(gl_decision.get("reason") or "")
                    if run_gl:
                        # Prefer the LLM's explicit query (single lookup);
                        # otherwise fall back to looping the candidates.
                        targets = [gl_query] if gl_query else candidates
                        for cond in targets:
                            hit = self.guideline_lookup_fn(cond) or {"match": None}
                            guideline_matches.append({"queried": cond, **hit})
                        n_matched = sum(
                            1 for g in guideline_matches if g.get("match") is not None
                        )
                        outcome = f"{n_matched}/{len(targets)} matched"
                    else:
                        outcome = "skipped by LLM"
                    if gl_reason:
                        outcome = f"{outcome} | reasoning: {gl_reason}"
                    self.log_action(
                        "guideline_lookup",
                        {
                            "n_candidates": len(candidates),
                            "ran":          run_gl,
                            "override_query": gl_query,
                        },
                        outcome=outcome,
                    )
                elif action == "summarize_findings":
                    if top_evidence:
                        mean_score = sum(e["score"] for e in top_evidence) / len(top_evidence)
                        retrieval_conf = max(0.0, min(1.0, mean_score))
                    else:
                        retrieval_conf = 0.0

                    findings = {
                        "retrieved_docs":       retrieved_docs,
                        "top_evidence":         top_evidence,
                        "guideline_matches":    guideline_matches,
                        "retrieval_confidence": round(retrieval_conf, 4),
                    }
                    # Wrap memory writes + message in `traced_action` so each
                    # diff links to the summarize_findings event (otherwise
                    # writes fire with no event in context and land
                    # "outside_traced_action").
                    with self.traced_action(
                        "summarize_findings",
                        {"keys": list(findings.keys())},
                        outcome="memory+message written",
                    ):
                        for k, v in findings.items():
                            self.memory[k] = v

                        self.send_message(
                            receiver=self.orchestrator_id,
                            message_type="finding",
                            content={
                                "n_docs":               len(retrieved_docs),
                                "top_evidence":         top_evidence,
                                "guideline_matches":    guideline_matches,
                                "retrieval_confidence": findings["retrieval_confidence"],
                            },
                        )

        return findings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_candidates(self, case: str) -> List[str]:
        """LLM-driven candidate-condition list; empty if no LLM."""
        if not case or self.llm is None:
            return []
        try:
            response = self.llm.invoke(_CONDITION_PROMPT.format(case=case))
        except Exception:
            return []
        return parse_json_list(extract_text(response))

    def _decide_evidence_tools(
        self,
        case: str,
        candidates: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Ask the LLM whether to run textbook_search and/or guideline_lookup,
        and what query string each should use.

        Returns a dict shaped like::

            {
                "textbook_search":  {"run": bool, "reason": str, "query": str},
                "guideline_lookup": {"run": bool, "reason": str, "query": str},
            }

        Falls back to "run both with default query" when no LLM is wired,
        the LLM raises, or the response can't be parsed as a JSON object —
        same default-on-failure stance as the SpecialistA gating helpers.
        """
        default = {
            "textbook_search":  {"run": True, "reason": "", "query": ""},
            "guideline_lookup": {"run": True, "reason": "", "query": ""},
        }
        if self.llm is None:
            return default
        try:
            response = self.llm.invoke(
                _EVIDENCE_DECISION_PROMPT.format(case=case, candidates=candidates),
            )
        except Exception:
            return default

        parsed = parse_json_object(extract_text(response))
        if not parsed:
            return default

        return {
            "textbook_search":  _coerce_decision(parsed.get("textbook_search")),
            "guideline_lookup": _coerce_decision(parsed.get("guideline_lookup")),
        }


def _coerce_decision(raw: Any) -> Dict[str, Any]:
    """Normalise one tool's decision sub-object; default to ``run: true``."""
    if not isinstance(raw, dict):
        return {"run": True, "reason": "", "query": ""}
    run_value = raw.get("run")
    return {
        "run":    run_value if isinstance(run_value, bool) else True,
        "reason": str(raw.get("reason", "") or ""),
        "query":  str(raw.get("query", "") or ""),
    }
