"""Pydantic schemas for multi-agent reasoning, citations, and grounded answers."""

from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentCitation(BaseModel):
    """Verifiable citation pointing to an ingested document passage."""
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    page_number: int = Field(..., description="Page number of evidence")
    chunk_index: int = Field(..., description="Chunk index")
    snippet: str = Field(..., description="Exact supporting snippet excerpt")

    model_config = {"from_attributes": True}


class WebCitation(BaseModel):
    """Verifiable citation pointing to an external web source."""
    title: str = Field(..., description="Page title")
    url: str = Field(..., description="Full URL of web source")
    snippet: str = Field(..., description="Extracted web content excerpt")

    model_config = {"from_attributes": True}


class AgentThoughtStep(BaseModel):
    """Granular audit trail step taken by an agent during reasoning."""
    agent_name: str = Field(..., description="Name of agent node: ResearchEvidenceAgent, WebSearchAgent, AnswerAgent")
    action: str = Field(..., description="Action taken: RETRIEVE, EVALUATE_EVIDENCE, WEB_SEARCH, SYNTHESIZE, VERIFY")
    rationale: str = Field(..., description="Explanation of why action was taken")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


    model_config = {"from_attributes": True}


class QuestionRequest(BaseModel):
    """Incoming user question request."""
    question: str = Field(..., min_length=1, description="Question text")
    document_id: Optional[str] = Field(None, description="Optional document filter")
    top_k: int = Field(5, ge=1, le=10, description="Max context chunks to retrieve")
    allow_web_search: bool = Field(True, description="Enable web search fallback if document evidence is insufficient")


class AnswerResponse(BaseModel):
    """Final grounded answer response with verified citations and thought trace."""
    question: str
    answer: str
    source_type: str = Field(..., description="Source classification: DOCUMENTS, WEB, HYBRID, INSUFFICIENT_EVIDENCE")
    confidence_score: float = Field(..., description="Confidence score of answer grounding (0.0 to 1.0)")
    document_citations: List[DocumentCitation] = Field(default_factory=list)
    web_citations: List[WebCitation] = Field(default_factory=list)
    thought_steps: List[AgentThoughtStep] = Field(default_factory=list)
    latency_ms: float = Field(0.0, description="Total execution time in milliseconds")
