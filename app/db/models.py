"""SQLAlchemy 2.0 database models for documents, task tracking, pages, and query audits."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for all database entities."""
    pass


def get_utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class Document(Base):
    """Stored document metadata and processing state."""
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False, default="UNKNOWN")
    minio_raw_path: Mapped[str] = mapped_column(String(500), nullable=False)
    minio_ocr_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    ocr_engine_used: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    ocr_text_preview: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ocr_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_pages: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(30), index=True, nullable=False, default="PENDING")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    task_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=get_utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=get_utc_now, onupdate=get_utc_now, nullable=False
    )

    # Relationships
    task_logs: Mapped[List["TaskLog"]] = relationship(
        "TaskLog", back_populates="document", cascade="all, delete-orphan"
    )
    pages: Mapped[List["DocumentPage"]] = relationship(
        "DocumentPage", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename='{self.filename}' status='{self.status}'>"


class DocumentPage(Base):
    """Detailed page-level OCR and extraction artifact metadata."""
    __tablename__ = "document_pages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    has_selectable_text: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    ocr_engine: Mapped[str] = mapped_column(String(50), nullable=False, default="NATIVE")
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    table_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    image_minio_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    structured_json_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    warnings: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=get_utc_now, nullable=False)

    # Relationships
    document: Mapped[Document] = relationship("Document", back_populates="pages")

    def __repr__(self) -> str:
        return f"<DocumentPage doc={self.document_id} page={self.page_number} engine={self.ocr_engine}>"


class TaskLog(Base):
    """Step-by-step progress logging for background Celery tasks."""
    __tablename__ = "task_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False
    )
    stage: Mapped[str] = mapped_column(String(50), nullable=False)
    progress: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=get_utc_now, nullable=False)

    # Relationships
    document: Mapped[Document] = relationship("Document", back_populates="task_logs")

    def __repr__(self) -> str:
        return f"<TaskLog task_id={self.task_id} stage='{self.stage}' progress={self.progress}%>"


class QueryAudit(Base):
    """Audit trail for RAG questions, retrieval routing, and LLM responses."""
    __tablename__ = "query_audits"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    routed_source: Mapped[str] = mapped_column(String(50), nullable=False)
    top_confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    threshold_used: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    response_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    llm_model: Mapped[str] = mapped_column(String(100), nullable=False)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    chunks_retrieved: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=get_utc_now, index=True, nullable=False)

    def __repr__(self) -> str:
        return f"<QueryAudit id={self.id} source='{self.routed_source}'>"
