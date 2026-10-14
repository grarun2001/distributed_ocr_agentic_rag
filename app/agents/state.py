"""LangGraph workflow state schema for multi-agent reasoning."""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from app.schemas.answer import AgentThoughtStep, DocumentCitation, WebCitation
from app.schemas.retrieval import RetrievalHit


class AgentWorkflowState(TypedDict):
    """Complete mutable state passed between LangGraph agent nodes."""
    question: str
    document_id: Optional[str]
    top_k: int
    allow_web_search: bool
    retrieved_chunks: List[RetrievalHit]
    evidence_is_sufficient: bool
    evidence_score: float
    evidence_summary: str
    web_search_needed: bool
    web_results: List[Dict[str, str]]
    answer: str
    document_citations: List[DocumentCitation]
    web_citations: List[WebCitation]
    thought_steps: List[AgentThoughtStep]
    source_type: str
    confidence_score: float
    iteration_count: int
    error: Optional[str]
