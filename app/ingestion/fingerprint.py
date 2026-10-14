"""Document SHA-256 fingerprinting utility."""

import hashlib
from typing import BinaryIO


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest for in-memory byte buffer."""
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def compute_sha256_stream(stream: BinaryIO, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hex digest for a file stream with bounded memory usage."""
    hasher = hashlib.sha256()
    pos = stream.tell()
    try:
        stream.seek(0)
        while chunk := stream.read(chunk_size):
            hasher.update(chunk)
        return hasher.hexdigest()
    finally:
        stream.seek(pos)
