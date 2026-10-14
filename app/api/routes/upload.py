"""File upload endpoints supporting single and batch document ingestion."""

from typing import List
from fastapi import APIRouter, Depends, File, UploadFile, status

from app.api.dependencies import get_upload_service
from app.schemas.upload import UploadResponse, BatchUploadResponse
from app.services.upload_service import UploadService

router = APIRouter(prefix="/upload", tags=["Upload & Ingestion"])


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a single document",
    description="Validates, fingerprints (SHA-256), deduplicates, and stores raw file in MinIO and metadata in PostgreSQL."
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (PDF, Image, Office doc, etc.)"),
    service: UploadService = Depends(get_upload_service)
) -> UploadResponse:
    """Ingest a single document into the pipeline."""
    return await service.process_upload(file)


@router.post(
    "/batch",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload multiple documents in a batch",
    description="Ingests multiple files concurrently, returning metadata and duplicate detection outcomes."
)
async def upload_documents_batch(
    files: List[UploadFile] = File(..., description="Multiple document files to upload"),
    service: UploadService = Depends(get_upload_service)
) -> BatchUploadResponse:
    """Ingest a batch of documents into the pipeline."""
    return await service.process_batch_upload(files)
