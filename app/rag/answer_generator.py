"""End-to-end question answering engine coordinating LangGraph agent execution."""

import time
import uuid
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import agentic_rag_graph
from app.agents.guardrails import detect_prompt_injection
from app.agents.state import AgentWorkflowState
from app.core.exceptions import AgentExecutionError
from app.core.logging import get_logger
from app.db.models import QueryAudit
from app.db.repositories import QueryAuditRepository
from app.schemas.answer import AnswerResponse, QuestionRequest

logger = get_logger("answer_generator")


class AnswerGenerator:
    """Coordinates multi-agent reasoning, guardrails, and citation verification."""

    def __init__(self, session: Optional[AsyncSession] = None) -> None:
        self.session = session
        self.audit_repo = QueryAuditRepository(session) if session else None

    async def answer_question(self, request: QuestionRequest) -> AnswerResponse:
        """Run complete LangGraph multi-agent flow on user question."""
        start_time = time.time()
        logger.info(f"Processing question via LangGraph: '{request.question}'...")

        # 1. Guardrail: Prompt Injection Detection
        is_injection, trigger = detect_prompt_injection(request.question)
        if is_injection:
            logger.warning(f"Prompt injection pattern detected: '{trigger}'. Refusing execution.")
            return AnswerResponse(
                question=request.question,
                answer="I cannot fulfill this request because it contains prompt injection patterns.",
                source_type="INSUFFICIENT_EVIDENCE",
                confidence_score=0.0,
                document_citations=[],
                web_citations=[],
                thought_steps=[],
                latency_ms=round((time.time() - start_time) * 1000, 2)
            )

        # 2. Initialize LangGraph State
        initial_state: AgentWorkflowState = {
            "question": request.question,
            "document_id": request.document_id,
            "top_k": request.top_k,
            "allow_web_search": request.allow_web_search,
            "retrieved_chunks": [],
            "evidence_is_sufficient": False,
            "evidence_score": 0.0,
            "evidence_summary": "",
            "web_search_needed": False,
            "web_results": [],
            "answer": "",
            "document_citations": [],
            "web_citations": [],
            "thought_steps": [],
            "source_type": "INSUFFICIENT_EVIDENCE",
            "confidence_score": 0.0,
            "iteration_count": 0,
            "error": None
        }

        # 3. Execute LangGraph State Machine
        try:
            final_state = agentic_rag_graph.invoke(initial_state)
        except Exception as exc:
            logger.error(f"LangGraph execution failed: {exc}", exc_info=True)
            raise AgentExecutionError(f"Multi-agent execution failure: {exc}")

        latency_ms = round((time.time() - start_time) * 1000, 2)

        # 4. Record Audit Log in Database
        if self.audit_repo and self.session:
            try:
                audit = QueryAudit(
                    id=str(uuid.uuid4()),
                    query_text=request.question,
                    routed_source=final_state.get("source_type", "UNKNOWN"),
                    top_confidence_score=final_state.get("confidence_score", 0.0),
                    threshold_used=0.40,
                    response_text=final_state.get("answer", "")[:1000],
                    llm_model="groq_llama_3.1",
                    latency_ms=latency_ms,
                    chunks_retrieved=len(final_state.get("retrieved_chunks", []))
                )
                await self.audit_repo.record_audit(audit)
                await self.session.commit()
            except Exception as audit_exc:
                logger.warning(f"Could not record audit log: {audit_exc}")

        return AnswerResponse(
            question=request.question,
            answer=final_state.get("answer", "No answer generated."),
            source_type=final_state.get("source_type", "INSUFFICIENT_EVIDENCE"),
            confidence_score=final_state.get("confidence_score", 0.0),
            document_citations=final_state.get("document_citations", []),
            web_citations=final_state.get("web_citations", []),
            thought_steps=final_state.get("thought_steps", []),
            latency_ms=latency_ms
        )
