"""Domain-specific exception hierarchy for the Distributed OCR Agentic RAG system."""

from typing import Any, Dict, Optional


class AppException(Exception):
    """Base exception for all domain-specific errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_SERVER_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}


class FileValidationError(AppException):
    """Raised when uploaded file fails size, MIME, or magic signature checks."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="FILE_VALIDATION_ERROR",
            status_code=400,
            details=details
        )


class StorageError(AppException):
    """Raised when MinIO object storage read/write operation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="STORAGE_ERROR",
            status_code=502,
            details=details
        )


class DatabaseError(AppException):
    """Raised when PostgreSQL queries or transactions fail."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=500,
            details=details
        )


class DocumentNotFoundError(AppException):
    """Raised when requested document or task ID is not found."""

    def __init__(self, document_id: str) -> None:
        super().__init__(
            message=f"Document with ID '{document_id}' was not found.",
            error_code="DOCUMENT_NOT_FOUND",
            status_code=404,
            details={"document_id": document_id}
        )


class TaskNotFoundError(AppException):
    """Raised when requested task ID is not found."""

    def __init__(self, task_id: str) -> None:
        super().__init__(
            message=f"Task with ID '{task_id}' was not found.",
            error_code="TASK_NOT_FOUND",
            status_code=404,
            details={"task_id": task_id}
        )


class OCRError(AppException):
    """Raised when OCR or document extraction pipeline fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="OCR_PROCESSING_ERROR",
            status_code=500,
            details=details
        )


class PreprocessingError(AppException):
    """Raised when image preprocessing or page deskew fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="PREPROCESSING_ERROR",
            status_code=500,
            details=details
        )


class IndexingError(AppException):
    """Raised when vector embedding generation or Milvus/BM25 insertion fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="INDEXING_ERROR",
            status_code=500,
            details=details
        )


class RetrievalError(AppException):
    """Raised during hybrid search or fusion."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="RETRIEVAL_ERROR",
            status_code=500,
            details=details
        )


class RerankerError(AppException):
    """Raised when cross-encoder reranker inference fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="RERANKER_ERROR",
            status_code=500,
            details=details
        )


class AgentExecutionError(AppException):
    """Raised during LangGraph agent execution or Groq LLM invocation."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="AGENT_EXECUTION_ERROR",
            status_code=502,
            details=details
        )


class CircuitBreakerOpenError(AppException):
    """Raised when a circuit breaker blocks requests to a degraded dependency."""

    def __init__(self, service_name: str) -> None:
        super().__init__(
            message=f"Circuit breaker is OPEN for service '{service_name}'. Requests temporarily blocked.",
            error_code="CIRCUIT_BREAKER_OPEN",
            status_code=503,
            details={"service_name": service_name}
        )
