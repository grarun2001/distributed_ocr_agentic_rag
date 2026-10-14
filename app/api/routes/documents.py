"""Document metadata, artifact retrieval, and management endpoints."""

from typing import Dict
from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import get_document_service
from app.schemas.document import DocumentDetailResponse, DocumentListResponse
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all uploaded documents",
    description="Retrieve a paginated list of all documents stored in the system ordered by creation time descending."
)
async def list_documents(
    limit: int = Query(50, ge=1, le=100, description="Max documents to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    service: DocumentService = Depends(get_document_service)
) -> DocumentListResponse:
    """List documents with pagination."""
    return await service.list_documents(limit=limit, offset=offset)


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get document details by ID",
    description="Fetch document details including page extractions and secure presigned URLs to download raw files and OCR JSON."
)
async def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service)
) -> DocumentDetailResponse:
    """Fetch complete document metadata and download URLs."""
    return await service.get_document(document_id)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a document",
    description="Cascades deletion across PostgreSQL records, MinIO artifacts, and Milvus vector embeddings."
)
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service)
) -> Dict[str, str]:
    """Delete document and all associated artifacts."""
    await service.delete_document(document_id)
    return {"message": f"Document '{document_id}' and all dependent artifacts were successfully deleted."}
