"""Hybrid search API endpoints for dense vector, BM25, and reranked retrieval."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_db
from app.rag.answer_generator import AnswerGenerator
from app.schemas.answer import AnswerResponse, QuestionRequest
from app.schemas.retrieval import HybridSearchQuery, HybridSearchResponse
from app.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["RAG & Retrieval"])


@router.post(
    "",
    response_model=HybridSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute hybrid search with Cross-Encoder reranking",
    description="Fuses Milvus dense vector search and persistent BM25 keyword search using RRF and neural Cross-Encoder reranking."
)
async def hybrid_search(
    request: HybridSearchQuery,
    session: AsyncSession = Depends(get_async_db)
) -> HybridSearchResponse:
    """Perform multi-stage hybrid search across ingested document chunks."""
    service = SearchService(session)
    return await service.execute_search(request)


@router.post(
    "/ask",
    response_model=AnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Agentic question answering with LangGraph multi-agent flow",
    description="Answers user queries using ResearchEvidenceAgent, fallback WebSearchAgent, and grounded AnswerVerificationAgent with verified citations."
)
async def ask_question(
    request: QuestionRequest,
    session: AsyncSession = Depends(get_async_db)
) -> AnswerResponse:
    """Answer question with verified document/web citations and agent reasoning trace."""
    generator = AnswerGenerator(session)
    return await generator.answer_question(request)

