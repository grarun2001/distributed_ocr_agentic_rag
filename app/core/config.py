"""Centralized system configuration using Pydantic Settings."""

import os
from functools import lru_cache
from typing import List, Union
from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable fallback."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application Information
    APP_NAME: str = "Distributed OCR Agentic RAG"
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    SECRET_KEY: str = "change-me-to-a-secure-random-secret-key"
    API_V1_STR: str = "/api/v1"
    CORS_ORIGINS: List[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000"
    ]

    # PostgreSQL Database
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "ocr_rag_db"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5433
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/ocr_rag_db"
    DATABASE_URL_SYNC: str = "postgresql://postgres:postgres@localhost:5433/ocr_rag_db"

    # Redis & Celery Task Queue
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # MinIO Object Storage
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ROOT_USER: str = "minioadmin"
    MINIO_ROOT_PASSWORD: str = "minioadmin"
    MINIO_USE_SSL: bool = False
    MINIO_BUCKET_DOCUMENTS: str = "ocr-documents"
    MINIO_BUCKET_ARTIFACTS: str = "ocr-artifacts"
    MINIO_BUCKET_EXPORT: str = "ocr-exports"

    # Milvus Vector Database
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_USER: str = ""
    MILVUS_PASSWORD: str = ""
    MILVUS_COLLECTION_NAME: str = "document_chunks"
    MILVUS_EMBEDDING_DIM: int = 384

    # Persistent BM25 Directory
    BM25_STORAGE_DIR: str = "data/bm25_indices"

    # Embedding & Cross-Encoder Models
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # OCR Engines
    TESSERACT_CMD: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    TESSERACT_LANGUAGES: str = "eng"
    ENABLE_PADDLEOCR: bool = False
    PADDLEOCR_USE_ANGLE_CLS: bool = True
    PADDLEOCR_LANG: str = "en"
    OCR_CONFIDENCE_THRESHOLD: float = 60.0
    OCR_ENGINE_DEFAULT: str = "hybrid"

    # Agentic AI & LLMs (Groq & Google Gemini)
    GROQ_API_KEY: str = ""
    GROQ_MODEL_NAME: str = "llama-3.1-70b-versatile"
    GROQ_TEMPERATURE: float = 0.1
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
    MAX_AGENT_ITERATIONS: int = 5

    # Web Search Fallback
    WEB_SEARCH_MAX_RESULTS: int = 5
    WEB_SEARCH_TIMEOUT_SECONDS: int = 10

    # System Safeguards & Limits
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = [
        "pdf", "png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp",
        "docx", "doc", "txt", "rtf", "md", "html",
        "xlsx", "xls", "csv",
        "pptx", "ppt", "xml"
    ]
    CIRCUIT_BREAKER_FAIL_MAX: int = 5
    CIRCUIT_BREAKER_RESET_TIMEOUT: int = 30

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton instance of system settings."""
    return Settings()
