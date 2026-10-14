"""Initialize MinIO buckets required by the distributed OCR RAG pipeline."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.storage.minio_client import get_minio_client


def main() -> int:
    print("=" * 60)
    print(" INITIALIZING MINIO S3 STORAGE BUCKETS")
    print("=" * 60)

    settings = get_settings()
    client = get_minio_client()

    required_buckets = [
        settings.MINIO_BUCKET_DOCUMENTS,
        settings.MINIO_BUCKET_ARTIFACTS,
        settings.MINIO_BUCKET_EXPORT
    ]

    for bucket in required_buckets:
        print(f"[*] Checking bucket: '{bucket}'...")
        client.ensure_bucket_exists(bucket)
        print(f"    --> Bucket '{bucket}' is ready.")

    print("\nAll MinIO buckets successfully initialized.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
