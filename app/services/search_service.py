"""Search service coordinating hybrid retrieval and audit trail recording."""

import time
import uuid
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import QueryAudit
from app.db.repositories import QueryAuditRepository
from app.rag.hybrid_search import get_search_engine
from app.schemas.retrieval import HybridSearchQuery, HybridSearchResponse

settings = get_settings()


class SearchService:
    """Provides high-level search execution and metrics collection."""

    def __init__(self, session: Optional[AsyncSession] = None) -> None:
        self.session = session
        self.engine = get_search_engine()
        self.audit_repo = QueryAuditRepository(session) if session else None

    async def execute_search(self, request: HybridSearchQuery) -> HybridSearchResponse:
        """Run hybrid search, measure latency, and record audit log."""
        start_time = time.time()

        hits = self.engine.search(
            query=request.query,
            top_k=request.top_k,
            filter_document_id=request.document_id,
            mode=request.mode
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)
        top_score = hits[0].score if hits else 0.0

        # Record audit log if session is active
        if self.audit_repo and self.session:
            try:
                audit = QueryAudit(
                    id=str(uuid.uuid4()),
                    query_text=request.query,
                    routed_source="HYBRID_SEARCH",
                    top_confidence_score=top_score,
                    threshold_used=0.0,
                    response_text=f"Retrieved {len(hits)} passages",
                    llm_model="none",
                    latency_ms=latency_ms,
                    chunks_retrieved=len(hits)
                )
                await self.audit_repo.record_audit(audit)
                await self.session.commit()
            except Exception:
                pass

        return HybridSearchResponse(
            query=request.query,
            total_hits=len(hits),
            mode=request.mode,
            hits=hits
        )
