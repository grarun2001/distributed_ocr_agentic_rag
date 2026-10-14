"""Research & Evidence Agent evaluating document evidence sufficiency and relevance."""

import re
from typing import List, Tuple
from app.core.logging import get_logger
from app.schemas.retrieval import RetrievalHit

logger = get_logger("evidence_checker")

REALTIME_TRIGGERS = [
    r"\btoday\b", r"\bcurrent(ly)?\b", r"\blatest\b", r"\bweather\b",
    r"\bstock price\b", r"\blive\b", r"\brecent news\b"
]


class ResearchEvidenceAgent:
    """Understands question intent, evaluates retrieved evidence, and decides if web search is needed."""

    def evaluate_evidence(
        self,
        question: str,
        retrieved_chunks: List[RetrievalHit]
    ) -> Tuple[bool, float, str, bool]:
        """
        Evaluate if document evidence is sufficient to answer the question.
        Returns:
            Tuple of (is_sufficient, evidence_score, rationale, web_search_needed)
        """
        # 1. Check for real-time / live knowledge triggers
        for pattern in REALTIME_TRIGGERS:
            if re.search(pattern, question, re.IGNORECASE):
                logger.info(f"Question '{question}' triggered real-time intent: '{pattern}'.")
                return (
                    False,
                    0.20,
                    f"Question requires real-time or live external information matching '{pattern}'.",
                    True
                )

        # 2. Check if no chunks were retrieved
        if not retrieved_chunks:
            logger.info("No document chunks retrieved for query.")
            return False, 0.0, "No document passages were matched in Milvus or BM25.", True

        # 3. Assess relevance score of top candidates
        top_chunk = retrieved_chunks[0]
        top_score = top_chunk.score
        is_reranked = (getattr(top_chunk, "retrieval_source", "") == "RERANKED")

        # 4. Token overlap heuristic
        query_words = set(re.findall(r"\b\w{3,}\b", question.lower()))
        # Remove common stop words
        stop_words = {
            "what", "when", "where", "which", "who", "whom", "this", "that", "from",
            "with", "have", "been", "tell", "show", "give", "please", "about", "does",
            "the", "and", "for"
        }
        keywords = query_words - stop_words

        matched_keywords = set()
        for chunk in retrieved_chunks[:3]:
            chunk_words = set(re.findall(r"\b\w{3,}\b", chunk.text.lower()))
            matched_keywords.update(keywords.intersection(chunk_words))

        keyword_coverage = (len(matched_keywords) / len(keywords)) if keywords else 1.0

        # Sufficiency heuristic:
        # If Cross-Encoder reranker was used:
        # A genuine semantic match has score >= 0.55 (strong) or score >= 0.20 with >= 50% keyword coverage.
        # Irrelevant passages have score < 0.10 (e.g. 0.0019) and must NOT be marked sufficient.
        if is_reranked:
            is_sufficient = (top_score >= 0.55) or (top_score >= 0.20 and keyword_coverage >= 0.50)
        else:
            # Fallback for pure RRF/BM25: requires majority keyword match and positive score
            is_sufficient = (keyword_coverage >= 0.60) and (top_score >= 0.01)

        if is_sufficient:
            rationale = (
                f"Sufficient evidence found in documents. Top score: {top_score:.4f}, "
                f"Keyword coverage: {keyword_coverage * 100:.0f}%."
            )
            logger.info(f"Evidence SUFFICIENT: {rationale}")
            return True, top_score, rationale, False
        else:
            rationale = (
                f"Document evidence is insufficient (Score: {top_score:.4f}, "
                f"Keyword coverage: {keyword_coverage * 100:.0f}%). Web search required."
            )
            logger.info(f"Evidence INSUFFICIENT: {rationale}")
            return False, top_score, rationale, True


_evidence_agent_instance = ResearchEvidenceAgent()


def get_evidence_agent() -> ResearchEvidenceAgent:
    """Retrieve singleton instance of ResearchEvidenceAgent."""
    return _evidence_agent_instance
