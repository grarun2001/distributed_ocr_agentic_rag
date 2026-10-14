# Distributed OCR + Agentic RAG System

A production-grade, distributed document ingestion, OCR extraction, hybrid RAG (Milvus + BM25), and LangGraph multi-agent AI system with Streamlit UI.

## Architecture Highlights
- **Distributed Ingestion & Queue:** FastAPI, MinIO S3 Object Storage, PostgreSQL, Redis, Celery Workers
- **Multi-Format Extraction & OCR:** PyMuPDF digital extraction, OpenCV image preprocessing, Tesseract OCR, and PaddleOCR
- **Hybrid Retrieval & Reranking:** Dense Milvus vector search + Sparse Persistent BM25 search fused with Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking
- **Multi-Agent Reasoning:** LangGraph workflow with Research & Evidence Agent, Answer & Verification Agent, and DuckDuckGo Web Search fallback
- **High-Aesthetic UI:** Streamlit application communicating strictly via FastAPI REST APIs

## Quickstart (Phase 1)
Verify dependencies and services:
```powershell
python scripts\verify_services.py
```
