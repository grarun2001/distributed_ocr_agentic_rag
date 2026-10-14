"""Security and cryptographic utility functions."""

import hashlib
from typing import BinaryIO, Union
from pathlib import Path


def calculate_sha256_bytes(data: bytes) -> str:
    """Calculate SHA-256 checksum of raw byte content."""
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def calculate_sha256_stream(file_obj: BinaryIO, chunk_size: int = 65536) -> str:
    """Calculate SHA-256 checksum of a file-like stream in chunks to preserve memory."""
    hasher = hashlib.sha256()
    original_position = file_obj.tell()
    try:
        file_obj.seek(0)
        while chunk := file_obj.read(chunk_size):
            hasher.update(chunk)
        return hasher.hexdigest()
    finally:
        file_obj.seek(original_position)


def calculate_sha256_file(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Calculate SHA-256 checksum of a file on local filesystem."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and illegal characters."""
    clean_name = Path(filename).name
    # Remove control characters and problematic Windows/POSIX symbols
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|', '\0']:
        clean_name = clean_name.replace(ch, '_')
    return clean_name.strip()
