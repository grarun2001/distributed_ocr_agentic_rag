"""Unit tests for SHA-256 fingerprinting utilities."""

import io
import hashlib
from app.ingestion.fingerprint import compute_sha256, compute_sha256_stream


def test_compute_sha256_bytes():
    """Verify SHA-256 computation matches standard hashlib for bytes."""
    data = b"Hello, Distributed OCR RAG!"
    expected = hashlib.sha256(data).hexdigest()
    assert compute_sha256(data) == expected


def test_compute_sha256_stream():
    """Verify SHA-256 computation on stream and check pointer preservation."""
    data = b"Streaming data for fingerprinting test."
    stream = io.BytesIO(data)
    stream.seek(5)
    expected = hashlib.sha256(data).hexdigest()

    result = compute_sha256_stream(stream, chunk_size=8)
    assert result == expected
    assert stream.tell() == 5  # stream position restored
