"""High-level document and OCR artifact storage manager."""

import json
from typing import Any, Dict, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.storage.minio_client import get_minio_client
from app.storage.object_keys import (
    get_raw_document_key,
    get_ocr_json_key,
    get_page_image_key,
    get_page_json_key,
    get_table_csv_key
)

settings = get_settings()
logger = get_logger("artifact_store")


class ArtifactStore:
    """Manages reading and writing document artifacts to object storage."""

    def __init__(self) -> None:
        self.minio = get_minio_client()
        self.doc_bucket = settings.MINIO_BUCKET_DOCUMENTS
        self.artifact_bucket = settings.MINIO_BUCKET_ARTIFACTS

    def save_raw_document(
        self,
        document_id: str,
        filename: str,
        data: bytes,
        content_type: str = "application/octet-stream"
    ) -> str:
        """Store original uploaded file in the documents bucket."""
        key = get_raw_document_key(document_id, filename)
        return self.minio.upload_bytes(
            bucket_name=self.doc_bucket,
            object_key=key,
            data=data,
            content_type=content_type
        )

    def get_raw_document(self, object_key: str) -> bytes:
        """Retrieve original raw document content."""
        return self.minio.download_bytes(self.doc_bucket, object_key)

    def save_ocr_result(self, document_id: str, result_dict: Dict[str, Any]) -> str:
        """Store structured full OCR JSON output in the artifacts bucket."""
        key = get_ocr_json_key(document_id)
        json_data = json.dumps(result_dict, indent=2, ensure_ascii=False).encode("utf-8")
        return self.minio.upload_bytes(
            bucket_name=self.artifact_bucket,
            object_key=key,
            data=json_data,
            content_type="application/json"
        )

    def get_ocr_result(self, document_id: str) -> Dict[str, Any]:
        """Download and parse structured full OCR JSON result."""
        key = get_ocr_json_key(document_id)
        raw_bytes = self.minio.download_bytes(self.artifact_bucket, key)
        return json.loads(raw_bytes.decode("utf-8"))

    def save_page_image(self, document_id: str, page_number: int, image_bytes: bytes) -> str:
        """Store rendered or preprocessed page image."""
        key = get_page_image_key(document_id, page_number)
        return self.minio.upload_bytes(
            bucket_name=self.artifact_bucket,
            object_key=key,
            data=image_bytes,
            content_type="image/png"
        )

    def get_page_image(self, document_id: str, page_number: int) -> bytes:
        """Retrieve preprocessed page image."""
        key = get_page_image_key(document_id, page_number)
        return self.minio.download_bytes(self.artifact_bucket, key)

    def save_page_json(self, document_id: str, page_number: int, page_dict: Dict[str, Any]) -> str:
        """Store single-page OCR structured JSON."""
        key = get_page_json_key(document_id, page_number)
        data = json.dumps(page_dict, indent=2, ensure_ascii=False).encode("utf-8")
        return self.minio.upload_bytes(
            bucket_name=self.artifact_bucket,
            object_key=key,
            data=data,
            content_type="application/json"
        )

    def save_table_csv(self, document_id: str, page_number: int, table_index: int, csv_content: str) -> str:
        """Store extracted table as CSV."""
        key = get_table_csv_key(document_id, page_number, table_index)
        return self.minio.upload_bytes(
            bucket_name=self.artifact_bucket,
            object_key=key,
            data=csv_content.encode("utf-8"),
            content_type="text/csv"
        )


_artifact_store_instance: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    """Retrieve singleton instance of ArtifactStore."""
    global _artifact_store_instance
    if _artifact_store_instance is None:
        _artifact_store_instance = ArtifactStore()
    return _artifact_store_instance
