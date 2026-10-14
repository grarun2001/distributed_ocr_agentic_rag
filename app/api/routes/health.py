"""System health and dependency readiness probes."""

from datetime import datetime, timezone
from typing import Any, Dict
from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import redis

from app.core.config import get_settings
from app.db.session import get_async_db
from app.storage.minio_client import get_minio_client
from app.indexing.milvus_store import get_milvus_store

settings = get_settings()
router = APIRouter(tags=["Health & Probes"])


@router.get("/health", status_code=status.HTTP_200_OK, summary="Overall system health check")
async def health_check(session: AsyncSession = Depends(get_async_db)) -> Dict[str, Any]:
    """Verify health and connectivity of PostgreSQL, Redis, MinIO, and Milvus."""
    services: Dict[str, str] = {}
    is_healthy = True

    # 1. PostgreSQL check
    try:
        await session.execute(text("SELECT 1"))
        services["database"] = "connected"
    except Exception as exc:
        services["database"] = f"error: {str(exc)}"
        is_healthy = False

    # 2. Redis check
    try:
        r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)
        if r.ping():
            services["redis"] = "connected"
        else:
            services["redis"] = "unreachable"
            is_healthy = False
    except Exception as exc:
        services["redis"] = f"error: {str(exc)}"
        is_healthy = False

    # 3. MinIO check
    try:
        minio_client = get_minio_client()
        minio_client.client.list_buckets()
        services["minio"] = "connected"
    except Exception as exc:
        services["minio"] = f"error: {str(exc)}"
        is_healthy = False

    # 4. Milvus check
    try:
        milvus_store = get_milvus_store()
        milvus_store.ensure_collection()
        services["milvus"] = "connected"
    except Exception as exc:
        services["milvus"] = f"error: {str(exc)}"
        is_healthy = False

    return {
        "status": "healthy" if is_healthy else "degraded",
        "services": services,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
