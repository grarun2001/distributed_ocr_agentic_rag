# Deployment & Setup Guide (Windows / CPU-Only)

This guide provides step-by-step instructions to set up, configure, and execute the **Distributed OCR + Agentic RAG System** on Windows using Python 3.12.8 and Docker Desktop.

---

## 1. Prerequisites

1. **Operating System**: Windows 10/11 64-bit with PowerShell 5.1+ or PowerShell 7.
2. **Python**: Python 3.12.8 64-bit installed. Ensure `python` or `py -3.12` is accessible in PATH.
3. **Docker Desktop**: Installed and running with WSL2 backend.
4. **Tesseract OCR**: Tesseract 5.5 for Windows installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`.
5. **Groq API Key**: Optional for live Llama 3.1 inference (`GROQ_API_KEY` in `.env`). In absence of key, an extractive grounded fallback synthesis engine executes automatically.

---

## 2. Infrastructure Startup (Docker Compose)

Start the backing datastores (PostgreSQL 16, Redis 7, MinIO, Milvus 2.4, and Etcd) using Docker Compose:

```powershell
docker compose up -d
```

Verify that all containers are healthy:

```powershell
docker compose ps
```

Expected running containers:
- `ocr_rag_postgres` (Port `5433 -> 5432`)
- `ocr_rag_redis` (Port `6379 -> 6379`)
- `ocr_rag_minio` (Ports `9000` S3 API, `9001` Web Console)
- `ocr_rag_milvus` (Port `19530`)
- `ocr_rag_milvus_etcd` (Port `2379`)

---

## 3. Environment Configuration

Copy `.env.example` to `.env` and verify key settings:

```ini
APP_ENV=development
LOG_LEVEL=INFO

# PostgreSQL (Note host port 5433)
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/ocr_rag_db
DATABASE_URL_SYNC=postgresql://postgres:postgres@localhost:5433/ocr_rag_db

# Redis & Celery
REDIS_HOST=localhost
REDIS_PORT=6379
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# LLM & Embedding Settings
GROQ_API_KEY=your_groq_api_key_here
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## 4. Running the Distributed Application

To run the complete system, open three PowerShell windows:

### Terminal 1: Celery Distributed Worker
Windows requires the `solo` or `threads` pool (since `fork` is not supported on Windows):

```powershell
py -3.12 -m celery -A app.queue.celery_app worker -l INFO -P solo
```

### Terminal 2: FastAPI Gateway Server
```powershell
py -3.12 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Terminal 3: Streamlit Interactive UI
```powershell
py -3.12 -m streamlit run frontend/streamlit_app.py --server.port 8501
```
Access the dashboard at `http://localhost:8501`.

---

## 5. Verification & Testing

Run the phase verification scripts to validate each layer independently:

```powershell
# Phase 1: Core configuration and dependencies
py -3.12 scripts/verify_services.py

# Phase 2: PostgreSQL, MinIO, and Milvus connections
py -3.12 scripts/verify_phase2.py

# Phase 3: File validation and upload gateway
py -3.12 scripts/verify_phase3.py

# Phase 4: OCR, PyMuPDF, and Celery processing
py -3.12 scripts/verify_phase4.py

# Phase 5: Chunking, BGE embeddings, BM25, and neural reranker
py -3.12 scripts/verify_phase5.py

# Phase 6: LangGraph Multi-Agent RAG and citation grounding
py -3.12 scripts/verify_phase6.py

# Phase 7: Streamlit components and API integration
py -3.12 scripts/verify_phase7.py

# Phase 8: Circuit Breaker, Exponential Retry, and Accuracy Benchmarks
py -3.12 scripts/verify_phase8.py
```

Run the automated test suite (38 test cases including E2E):

```powershell
py -3.12 -m pytest tests/ -v
```

---

## 6. Troubleshooting Common Issues

### Issue 1: Port 5432 conflict with local PostgreSQL
- **Cause**: Local PostgreSQL service already running on port 5432.
- **Solution**: The Docker compose configuration maps internal `5432` to host port `5433`. Verify `.env` uses port `5433`.

### Issue 2: Celery Worker crashes on Windows with `ValueError: not enough values to unpack`
- **Cause**: Default Celery `prefork` pool is unsupported on Windows OS.
- **Solution**: Always launch Celery with the `-P solo` argument on Windows.

### Issue 3: `RuntimeError: Event loop is closed` on Windows with asyncpg
- **Cause**: Windows ProactorEventLoop closes when a synchronous context exits.
- **Solution**: Ensure in-process scripts execute within a unified `asyncio.run()` loop or use `await async_engine.dispose()`.
