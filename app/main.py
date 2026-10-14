"""FastAPI Gateway entrypoint for Distributed OCR Agentic RAG system."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.exception_handlers import app_exception_handler, generic_exception_handler
from app.api.routes import documents, health, search, tasks, upload
from app.core.config import get_settings
from app.core.exceptions import AppException
from app.core.logging import get_logger, setup_logging
from app.db.session import init_db_tables_async
from app.storage.minio_client import get_minio_client

settings = get_settings()
setup_logging(settings.LOG_LEVEL)
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle manager."""
    logger.info(f"Starting {settings.APP_NAME} in '{settings.APP_ENV}' mode...")

    # Ensure database schema is ready
    try:
        await init_db_tables_async()
        logger.info("PostgreSQL database tables verified.")
    except Exception as exc:
        logger.error(f"Failed ensuring database tables during startup: {exc}")

    # Ensure MinIO buckets are present
    try:
        minio_client = get_minio_client()
        minio_client.ensure_bucket_exists(settings.MINIO_BUCKET_DOCUMENTS)
        minio_client.ensure_bucket_exists(settings.MINIO_BUCKET_ARTIFACTS)
        minio_client.ensure_bucket_exists(settings.MINIO_BUCKET_EXPORT)
        logger.info("MinIO storage buckets verified.")
    except Exception as exc:
        logger.error(f"Failed ensuring MinIO buckets during startup: {exc}")

    yield

    logger.info(f"Shutting down {settings.APP_NAME}...")


def create_application() -> FastAPI:
    """Build and configure the FastAPI application instance."""
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        description="Production-grade Distributed OCR + RAG + Multi-Agent AI System",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # Enable CORS for Streamlit and local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Exception Handlers
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Mount Route Blueprints
    app.include_router(health.router, prefix="", tags=["Health"])
    app.include_router(health.router, prefix=settings.API_V1_STR, tags=["Health"])
    app.include_router(upload.router, prefix=settings.API_V1_STR)
    app.include_router(tasks.router, prefix=settings.API_V1_STR)
    app.include_router(documents.router, prefix=settings.API_V1_STR)
    app.include_router(search.router, prefix=settings.API_V1_STR)

    return app


app = create_application()
