"""HTTP client communicating exclusively with the FastAPI Gateway backend."""

import os
from typing import Any, Dict, List, Optional
import requests

DEFAULT_BASE_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8000/api/v1")


class APIClient:
    """Client wrapper for all Distributed OCR & Agentic RAG API endpoints."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def check_health(self) -> Dict[str, Any]:
        """Fetch system health and dependencies status."""
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                status_str = str(data.get("status", "healthy")).upper()
                services = data.get("services", {})
                return {
                    "status": status_str,
                    "dependencies": services,
                    "services": services,
                    "timestamp": data.get("timestamp")
                }
            return {"status": "UNHEALTHY", "dependencies": {}, "services": {}, "error": f"Status {resp.status_code}"}
        except Exception as exc:
            return {"status": "UNREACHABLE", "dependencies": {}, "services": {}, "error": str(exc)}


    def upload_file(self, file_bytes: bytes, filename: str, content_type: str = "application/octet-stream") -> Dict[str, Any]:
        """Upload a file to FastAPI gateway for ingestion and Celery dispatch."""
        files = {"file": (filename, file_bytes, content_type)}
        resp = self.session.post(f"{self.base_url}/upload", files=files, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Poll task processing status and milestone progress."""
        resp = self.session.get(f"{self.base_url}/tasks/{task_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def list_documents(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Fetch list of ingested documents with retry."""
        import time
        last_exc = None
        for attempt in range(3):
            try:
                resp = self.session.get(
                    f"{self.base_url}/documents",
                    params={"limit": limit, "offset": offset},
                    timeout=15
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(0.5)
        if last_exc:
            raise last_exc
        return {"total": 0, "items": []}

    def get_document(self, document_id: str) -> Dict[str, Any]:
        """Fetch complete document metadata, pages, and artifact URLs."""
        resp = self.session.get(f"{self.base_url}/documents/{document_id}", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete document, vector embeddings, and storage artifacts."""
        resp = self.session.delete(f"{self.base_url}/documents/{document_id}", timeout=20)
        resp.raise_for_status()
        return resp.json()

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
        mode: str = "HYBRID_RERANK"
    ) -> Dict[str, Any]:
        """Execute hybrid search across Milvus, BM25, and neural reranker."""
        payload = {
            "query": query,
            "top_k": top_k,
            "document_id": document_id,
            "mode": mode
        }
        resp = self.session.post(f"{self.base_url}/search", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def ask_question(
        self,
        question: str,
        document_id: Optional[str] = None,
        top_k: int = 5,
        allow_web_search: bool = True
    ) -> Dict[str, Any]:
        """Submit question to LangGraph multi-agent RAG workflow."""
        payload = {
            "question": question,
            "document_id": document_id,
            "top_k": top_k,
            "allow_web_search": allow_web_search
        }
        resp = self.session.post(f"{self.base_url}/search/ask", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()


_client_instance = APIClient()


def get_api_client() -> APIClient:
    """Retrieve singleton instance of APIClient."""
    return _client_instance
