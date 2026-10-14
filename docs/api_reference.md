# REST API Reference Manual

The Distributed OCR & Agentic RAG System exposes a production REST API via FastAPI at `http://localhost:8000/api/v1`.

---

## 1. Document Ingestion & Upload

### `POST /api/v1/upload`
Upload a document or image for asynchronous processing.

**Headers:**
- `Content-Type: multipart/form-data`

**Request Body:**
- `file`: Binary file contents (Max: 50MB). Supported formats: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.docx`, `.xlsx`, `.pptx`, `.rtf`, `.html`, `.txt`, `.csv`.

**Response (`201 Created`):**
```json
{
  "document_id": "45e85a4a-a61e-475b-b9c1-9189b781575a",
  "task_id": "task_45e85a4a",
  "filename": "annual_report.pdf",
  "file_size": 1048576,
  "content_type": "application/pdf",
  "category": "PDF_DIGITAL",
  "file_hash": "de5b1ce861ab9f8a63fdb6554e949e8de9477a4c49e6ab06b39e62b81bb03688",
  "status": "PENDING",
  "message": "File uploaded successfully. Processing task dispatched."
}
```

---

## 2. Background Task Monitoring

### `GET /api/v1/tasks/{task_id}`
Poll task progress and milestone lifecycle logs.

**Response (`200 OK`):**
```json
{
  "task_id": "task_45e85a4a",
  "document_id": "45e85a4a-a61e-475b-b9c1-9189b781575a",
  "status": "SUCCESS",
  "progress": 100.0,
  "current_stage": "COMPLETE",
  "error_message": null,
  "logs": [
    {
      "stage": "INGESTION",
      "progress": 5.0,
      "message": "File ingested and stored in MinIO.",
      "created_at": "2026-10-06T09:20:08.446574Z"
    },
    {
      "stage": "OCR_EXTRACTION",
      "progress": 60.0,
      "message": "OCR completed. Extracted 4 pages.",
      "created_at": "2026-10-06T09:20:12.100234Z"
    },
    {
      "stage": "INDEXING",
      "progress": 100.0,
      "message": "Document indexed into Milvus & BM25.",
      "created_at": "2026-10-06T09:20:15.892112Z"
    }
  ],
  "updated_at": "2026-10-06T09:20:15.892112Z"
}
```

---

## 3. Document Repository Management

### `GET /api/v1/documents`
List all uploaded documents with pagination.

**Query Parameters:**
- `limit`: integer (default `50`, max `100`)
- `offset`: integer (default `0`)

**Response (`200 OK`):**
```json
{
  "total": 12,
  "items": [
    {
      "id": "45e85a4a-a61e-475b-b9c1-9189b781575a",
      "filename": "annual_report.pdf",
      "file_size": 1048576,
      "content_type": "application/pdf",
      "category": "PDF_DIGITAL",
      "status": "SUCCESS",
      "ocr_engine_used": "NATIVE",
      "ocr_confidence": 100.0,
      "total_pages": 4,
      "total_chunks": 8,
      "ocr_text_preview": "Annual Report 2026...",
      "raw_file_url": "http://localhost:9000/documents/raw/...",
      "ocr_json_url": "http://localhost:9000/artifacts/ocr/...",
      "created_at": "2026-10-06T09:20:08.434474Z",
      "updated_at": "2026-10-06T09:20:15.892112Z"
    }
  ]
}
```

### `GET /api/v1/documents/{document_id}`
Fetch comprehensive document metadata and page breakdown.

### `DELETE /api/v1/documents/{document_id}`
Cascading deletion of a document across PostgreSQL, MinIO, and Milvus vector collections.

**Response (`200 OK`):**
```json
{
  "message": "Document '45e85a4a-a61e-475b-b9c1-9189b781575a' and all dependent artifacts were successfully deleted."
}
```

---

## 4. Hybrid Search & Retrieval

### `POST /api/v1/search`
Execute hybrid retrieval combining Milvus dense vector search and persistent BM25 sparse keyword matching with neural Cross-Encoder reranking.

**Request Body:**
```json
{
  "query": "What was the total revenue in Q3?",
  "top_k": 5,
  "document_id": null,
  "mode": "HYBRID_RERANK"
}
```
*Supported modes: `VECTOR_ONLY`, `BM25_ONLY`, `HYBRID_RRF`, `HYBRID_RERANK`*

**Response (`200 OK`):**
```json
{
  "query": "What was the total revenue in Q3?",
  "total_hits": 5,
  "mode": "HYBRID_RERANK",
  "hits": [
    {
      "chunk_id": "45e85a4a_2",
      "document_id": "45e85a4a-a61e-475b-b9c1-9189b781575a",
      "filename": "annual_report.pdf",
      "page_number": 3,
      "chunk_index": 2,
      "text": "Total revenue for the third quarter reached $45.2M, representing an increase of 14% year-over-year.",
      "score": 0.9412,
      "retrieval_source": "RERANKED"
    }
  ]
}
```

---

## 5. Agentic Multi-Agent RAG

### `POST /api/v1/search/ask`
Execute LangGraph multi-agent question answering with evidence sufficiency evaluation, DuckDuckGo web search fallback, and citation verification.

**Request Body:**
```json
{
  "question": "What is the maximum worker concurrency allowed in Project Hyperion?",
  "document_id": null,
  "top_k": 3,
  "allow_web_search": true
}
```

**Response (`200 OK`):**
```json
{
  "question": "What is the maximum worker concurrency allowed in Project Hyperion?",
  "answer": "Project Hyperion authorizes a maximum concurrency of 128 workers [Doc: hyperion_spec.pdf, Page: 1].",
  "source_type": "DOCUMENTS",
  "confidence_score": 0.95,
  "document_citations": [
    {
      "document_id": "45e85a4a-a61e-475b-b9c1-9189b781575a",
      "filename": "hyperion_spec.pdf",
      "page_number": 1,
      "chunk_index": 0,
      "snippet": "Authorized Maximum Concurrency: 128 workers."
    }
  ],
  "web_citations": [],
  "thought_steps": [
    {
      "agent_name": "ResearchEvidenceAgent",
      "action": "EVALUATE_EVIDENCE",
      "rationale": "Sufficient evidence found in documents. Top score: 0.99, Keyword coverage: 80%."
    },
    {
      "agent_name": "AnswerVerificationAgent",
      "action": "VERIFY_AND_SYNTHESIZE",
      "rationale": "Synthesized grounded response. Verified 1 document citation(s) and 0 web citation(s). Source: DOCUMENTS."
    }
  ],
  "latency_ms": 218.45
}
```

---

## 6. System Probes & Healthcheck

### `GET /api/v1/health`
Verify readiness and liveliness of all underlying infrastructure components.

**Response (`200 OK`):**
```json
{
  "status": "healthy",
  "services": {
    "database": "connected",
    "redis": "connected",
    "minio": "connected",
    "milvus": "connected"
  },
  "timestamp": "2026-10-06T09:22:35.123456Z"
}
```
