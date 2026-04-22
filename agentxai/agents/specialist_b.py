"""
Specialist B — Evidence Retriever.

Pipeline (one ``active_plan``):

1. ``extract_candidate_conditions`` — LLM produces a short list of plausible
                                       conditions to look up
2. ``pubmed_search``                — top-k textbook passages for the case
                                       (pubmed_search uses the textbook FAISS
                                       index as a substitute corpus)
3. ``guideline_lookup``             — fuzzy guideline match per candidate
4. ``summarize_findings``           — write four memory keys + send a
                                       ``finding`` message to the Orchestrator

Memory keys written:

    retrieved_docs        : List[dict]   # full pubmed_search response
    top_evidence          : List[dict]   # top-N (default 3) {doc_id, score, source_file, snippet}
    guideline_matches     : List[dict]   # one per candidate condition (match=None if missed)
    retrieval_confidence  : float        # mean score of top_evidence (clamped to [0, 1])
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from agentxai.agents._llm_utils import extract_text, parse_json_list
from agentxai.agents.base import TracedAgent, make_default_llm
from agentxai.tools.guideline_lookup import guideline_lookup as _real_guideline_lookup
from agentxai.tools.pubmed_search import pubmed_search as _real_pubmed_search


_CONDITION_PROMPT = (
    "You are a clinical reasoner. From the patient case below, list up to 5 "
    "plausible candidate conditions (concise diagnosis names) that should be "
    "looked up in clinical guidelines.\n"
    "Return ONLY a JSON array of strings. No prose, no markdown.\n\n"
    "Patient case:\n{case}"
)

_SNIPPET_CHARS = 240


class SpecialistB(TracedAgent):
    """Evidence retriever — pubmed_search + guideline_lookup + LLM candidate gen."""

    def __init__(
        self,
        *,
        agent_id: str = "specialist_b",
        pubmed_search_fn: Optional[Callable[..., List[dict]]] = None,
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
        self.pubmed_search_fn = pubmed_search_fn or _real_pubmed_search
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

        with self.active_plan([
            "extract_candidate_conditions",
            "pubmed_search",
            "guideline_lookup",
            "summarize_findings",
        ]):
            # 1. Candidate conditions
            candidates = self._extract_candidates(case)
            self.log_action(
                "extract_candidate_conditions",
                {"case_chars": len(case)},
                outcome=f"{len(candidates)} candidates",
            )

            # 2. PubMed (textbook) search
            retrieved_docs: List[dict] = list(self.pubmed_search_fn(case, k=self.k_docs) or [])
            top_evidence: List[dict] = [
                {
                    "doc_id":      d.get("doc_id", ""),
                    "score":       float(d.get("score", 0.0)),
                    "source_file": d.get("source_file", ""),
                    "snippet":     str(d.get("text", ""))[:_SNIPPET_CHARS],
                }
                for d in retrieved_docs[: self.top_evidence_n]
            ]
            self.log_action(
                "pubmed_search",
                {"query_chars": len(case), "k": self.k_docs},
                outcome=f"{len(retrieved_docs)} docs, top score "
                        f"{(top_evidence[0]['score'] if top_evidence else 0.0):.3f}",
            )

            # 3. Guideline lookup per candidate
            guideline_matches: List[dict] = []
            for cond in candidates:
                hit = self.guideline_lookup_fn(cond) or {"match": None}
                guideline_matches.append({"queried": cond, **hit})
            n_matched = sum(1 for g in guideline_matches if g.get("match") is not None)
            self.log_action(
                "guideline_lookup",
                {"n_candidates": len(candidates)},
                outcome=f"{n_matched}/{len(candidates)} matched",
            )

            # 4. Summarise → memory + finding message
            if top_evidence:
                mean_score = sum(e["score"] for e in top_evidence) / len(top_evidence)
                retrieval_conf = max(0.0, min(1.0, mean_score))
            else:
                retrieval_conf = 0.0

            findings: Dict[str, Any] = {
                "retrieved_docs":       retrieved_docs,
                "top_evidence":         top_evidence,
                "guideline_matches":    guideline_matches,
                "retrieval_confidence": round(retrieval_conf, 4),
            }
            for k, v in findings.items():
                self.memory[k] = v

            # The orchestrator only needs a slim summary, not the full doc bodies.
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
            self.log_action(
                "summarize_findings",
                {"keys": list(findings.keys())},
                outcome="memory+message written",
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
