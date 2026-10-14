"""Persistent disk-backed BM25 sparse keyword index store."""

import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from rank_bm25 import BM25Plus

from app.core.config import get_settings
from app.core.exceptions import IndexingError, RetrievalError
from app.core.logging import get_logger
from app.schemas.retrieval import DocumentChunkItem

settings = get_settings()
logger = get_logger("bm25_store")


def tokenize_bm25(text: str) -> List[str]:
    """Extract lowercase alphanumeric tokens for BM25 lexical matching."""
    return re.findall(r"\b\w+\b", text.lower())


class PersistentBM25Store:
    """Manages persistent disk serialization and BM25Plus scoring across app restarts."""

    def __init__(self, storage_dir: Optional[str] = None) -> None:
        self.storage_dir = Path(storage_dir or settings.BM25_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.storage_dir / "bm25_manifest.json"
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_all_indices()

    def _load_all_indices(self) -> None:
        """Load all persisted .pkl files from disk on initialization."""
        count = 0
        for pkl_file in self.storage_dir.glob("*.pkl"):
            doc_id = pkl_file.stem
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                    self._cache[doc_id] = data
                    count += 1
            except Exception as exc:
                logger.warning(f"Could not load BM25 index from '{pkl_file}': {exc}")
        logger.info(f"Loaded {count} persistent BM25 indices from '{self.storage_dir}'.")

    def index_document(self, document_id: str, chunks: List[DocumentChunkItem]) -> int:
        """Build BM25 index for document chunks and persist to disk as .pkl."""
        if not chunks:
            return 0

        try:
            tokenized_corpus = [tokenize_bm25(c.text) for c in chunks]
            bm25 = BM25Plus(tokenized_corpus)

            chunks_data = [
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.metadata.document_id,
                    "filename": c.metadata.filename,
                    "page_number": c.metadata.page_number,
                    "chunk_index": c.metadata.chunk_index,
                    "text": c.text,
                    "char_count": c.metadata.char_count,
                    "has_table": c.metadata.has_table
                }
                for c in chunks
            ]

            payload = {
                "document_id": document_id,
                "chunks": chunks_data,
                "tokenized_corpus": tokenized_corpus,
                "bm25": bm25
            }

            # 1. Save to disk
            file_path = self.storage_dir / f"{document_id}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(payload, f)

            # 2. Update memory cache
            self._cache[document_id] = payload

            # 3. Update manifest
            self._update_manifest(document_id, len(chunks))
            logger.info(f"Persisted BM25 index for doc '{document_id}' ({len(chunks)} chunks) to '{file_path}'.")
            return len(chunks)

        except Exception as exc:
            logger.error(f"Failed indexing BM25 for document '{document_id}': {exc}")
            raise IndexingError(f"BM25 persistence failure: {exc}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search across persistent BM25 indices and return ranked chunk hits."""
        tokenized_query = tokenize_bm25(query)
        if not tokenized_query:
            return []

        all_matches: List[Dict[str, Any]] = []

        target_doc_ids = [filter_document_id] if filter_document_id else list(self._cache.keys())

        for doc_id in target_doc_ids:
            if doc_id not in self._cache:
                # Attempt disk reload in case another worker updated it
                pkl_file = self.storage_dir / f"{doc_id}.pkl"
                if pkl_file.exists():
                    with open(pkl_file, "rb") as f:
                        self._cache[doc_id] = pickle.load(f)
                else:
                    continue

            doc_entry = self._cache[doc_id]
            bm25: BM25Okapi = doc_entry["bm25"]
            chunks: List[Dict[str, Any]] = doc_entry["chunks"]

            scores = bm25.get_scores(tokenized_query)
            q_set = set(tokenized_query)
            for idx, score in enumerate(scores):
                chunk_tokens = set(doc_entry["tokenized_corpus"][idx])
                if score > 0.0 and chunk_tokens.intersection(q_set):
                    c = chunks[idx]
                    all_matches.append({
                        "chunk_id": c["chunk_id"],
                        "document_id": c["document_id"],
                        "filename": c["filename"],
                        "page_number": c["page_number"],
                        "chunk_index": c["chunk_index"],
                        "text": c["text"],
                        "score": float(score),
                        "retrieval_source": "BM25"
                    })

        # Sort descending by raw BM25 score
        all_matches.sort(key=lambda x: x["score"], reverse=True)
        top_hits = all_matches[:top_k]

        # Normalize scores to 0.0 - 1.0 relative to max score
        if top_hits and top_hits[0]["score"] > 0:
            max_s = top_hits[0]["score"]
            for h in top_hits:
                h["score"] = round(h["score"] / max_s, 4)

        return top_hits

    def delete_document(self, document_id: str) -> bool:
        """Purge document index from memory and delete file from disk."""
        self._cache.pop(document_id, None)
        file_path = self.storage_dir / f"{document_id}.pkl"
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted persistent BM25 index file '{file_path}'.")
            return True
        return False

    def _update_manifest(self, document_id: str, chunk_count: int) -> None:
        """Record indexed document metadata in JSON manifest."""
        manifest: Dict[str, Any] = {}
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                manifest = {}

        manifest[document_id] = {"chunks": chunk_count}
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)


_bm25_store_instance: Optional[PersistentBM25Store] = None


def get_bm25_store() -> PersistentBM25Store:
    """Retrieve singleton instance of PersistentBM25Store."""
    global _bm25_store_instance
    if _bm25_store_instance is None:
        _bm25_store_instance = PersistentBM25Store()
    return _bm25_store_instance
