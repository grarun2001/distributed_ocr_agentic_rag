"""MinIO S3-compatible object storage client with retries and bucket helpers."""

import io
from datetime import timedelta
from typing import Optional
from minio import Minio
from minio.error import S3Error

from app.core.config import get_settings
from app.core.exceptions import StorageError
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger("minio_client")


class MinioClientWrapper:
    """Thread-safe client wrapper for MinIO operations."""

    def __init__(self) -> None:
        self.client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ROOT_USER,
            secret_key=settings.MINIO_ROOT_PASSWORD,
            secure=settings.MINIO_USE_SSL
        )

    def ensure_bucket_exists(self, bucket_name: str) -> None:
        """Create bucket if it does not already exist."""
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"Created MinIO bucket '{bucket_name}'.")
        except Exception as exc:
            logger.error(f"Failed to check/create bucket '{bucket_name}': {exc}")
            raise StorageError(f"MinIO bucket error for '{bucket_name}': {exc}")

    def upload_bytes(
        self,
        bucket_name: str,
        object_key: str,
        data: bytes,
        content_type: str = "application/octet-stream"
    ) -> str:
        """Upload raw bytes to MinIO object storage."""
        self.ensure_bucket_exists(bucket_name)
        stream = io.BytesIO(data)
        try:
            self.client.put_object(
                bucket_name=bucket_name,
                object_name=object_key,
                data=stream,
                length=len(data),
                content_type=content_type
            )
            return object_key
        except Exception as exc:
            logger.error(f"Failed uploading {object_key} to {bucket_name}: {exc}")
            raise StorageError(f"MinIO upload error for '{object_key}': {exc}")

    def upload_file(
        self,
        bucket_name: str,
        object_key: str,
        file_path: str,
        content_type: str = "application/octet-stream"
    ) -> str:
        """Upload local file to MinIO object storage."""
        self.ensure_bucket_exists(bucket_name)
        try:
            self.client.fput_object(
                bucket_name=bucket_name,
                object_name=object_key,
                file_path=file_path,
                content_type=content_type
            )
            return object_key
        except Exception as exc:
            logger.error(f"Failed uploading file {file_path} to {bucket_name}/{object_key}: {exc}")
            raise StorageError(f"MinIO file upload error for '{object_key}': {exc}")

    def download_bytes(self, bucket_name: str, object_key: str) -> bytes:
        """Download object content as raw bytes."""
        try:
            response = self.client.get_object(bucket_name, object_key)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as s3_err:
            if s3_err.code == "NoSuchKey":
                raise StorageError(f"Object '{object_key}' not found in bucket '{bucket_name}'")
            raise StorageError(f"S3 error downloading '{object_key}': {s3_err}")
        except Exception as exc:
            logger.error(f"Failed downloading {object_key} from {bucket_name}: {exc}")
            raise StorageError(f"MinIO download error for '{object_key}': {exc}")

    def download_file(self, bucket_name: str, object_key: str, local_path: str) -> str:
        """Download object directly to a local file path."""
        try:
            self.client.fget_object(bucket_name, object_key, local_path)
            return local_path
        except Exception as exc:
            logger.error(f"Failed downloading {object_key} to {local_path}: {exc}")
            raise StorageError(f"MinIO file download error for '{object_key}': {exc}")

    def object_exists(self, bucket_name: str, object_key: str) -> bool:
        """Check if an object exists in the specified bucket."""
        try:
            self.client.stat_object(bucket_name, object_key)
            return True
        except S3Error as err:
            if err.code == "NoSuchKey":
                return False
            raise StorageError(f"MinIO stat error for '{object_key}': {err}")
        except Exception:
            return False

    def get_presigned_url(self, bucket_name: str, object_key: str, expires_seconds: int = 3600) -> str:
        """Generate presigned GET URL for secure client downloads."""
        try:
            return self.client.presigned_get_object(
                bucket_name=bucket_name,
                object_name=object_key,
                expires=timedelta(seconds=expires_seconds)
            )
        except Exception as exc:
            logger.error(f"Failed generating presigned URL for {object_key}: {exc}")
            raise StorageError(f"MinIO presigned URL error for '{object_key}': {exc}")

    def delete_object(self, bucket_name: str, object_key: str) -> None:
        """Remove an object from storage."""
        try:
            self.client.remove_object(bucket_name, object_key)
        except Exception as exc:
            logger.error(f"Failed deleting {object_key} from {bucket_name}: {exc}")
            raise StorageError(f"MinIO delete error for '{object_key}': {exc}")


_minio_client_instance: Optional[MinioClientWrapper] = None


def get_minio_client() -> MinioClientWrapper:
    """Retrieve singleton instance of MinioClientWrapper."""
    global _minio_client_instance
    if _minio_client_instance is None:
        _minio_client_instance = MinioClientWrapper()
    return _minio_client_instance
