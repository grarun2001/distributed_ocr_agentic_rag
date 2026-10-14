"""Security and structural validation for uploaded files."""

from pathlib import Path
from typing import Optional
from app.core.config import get_settings
from app.core.constants import SUPPORTED_EXTENSIONS, MAGIC_SIGNATURES
from app.core.exceptions import FileValidationError
from app.ingestion.mime_detection import detect_mime_type

settings = get_settings()


class FileValidator:
    """Validates size, extension, MIME type, and binary magic bytes of incoming uploads."""

    def __init__(self) -> None:
        self.max_size_bytes = settings.max_upload_size_bytes
        self.allowed_extensions = set(settings.ALLOWED_EXTENSIONS)

    def validate(self, filename: str, file_size: int, header_bytes: Optional[bytes] = None) -> str:
        """Validate uploaded file parameters and return canonical MIME content type."""
        # 1. Check non-empty
        if file_size <= 0:
            raise FileValidationError(f"File '{filename}' is empty (0 bytes).")

        # 2. Check maximum size limit
        if file_size > self.max_size_bytes:
            raise FileValidationError(
                f"File '{filename}' exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB.",
                details={"file_size": file_size, "max_allowed": self.max_size_bytes}
            )

        # 3. Check extension
        ext = Path(filename).suffix.lstrip(".").lower()
        if not ext or ext not in self.allowed_extensions:
            raise FileValidationError(
                f"File format '.{ext}' is not supported.",
                details={"filename": filename, "extension": ext, "allowed": list(self.allowed_extensions)}
            )

        # 4. Check magic bytes for known formats
        if header_bytes and len(header_bytes) >= 4:
            if ext == "pdf" and not header_bytes.startswith(MAGIC_SIGNATURES["pdf"]):
                raise FileValidationError(f"Invalid PDF header in file '{filename}'.")
            elif ext == "png" and not header_bytes.startswith(MAGIC_SIGNATURES["png"]):
                raise FileValidationError(f"Invalid PNG signature in file '{filename}'.")
            elif ext in ["jpg", "jpeg"] and not header_bytes.startswith(b"\xff\xd8"):
                raise FileValidationError(f"Invalid JPEG signature in file '{filename}'.")
            elif ext in ["docx", "xlsx", "pptx"] and not header_bytes.startswith(MAGIC_SIGNATURES["zip_office"]):
                raise FileValidationError(f"Corrupt Office OpenXML container in file '{filename}'.")

        # 5. Detect and return valid MIME type
        detected_mime = detect_mime_type(filename, header_bytes)
        return detected_mime


_file_validator_instance: Optional[FileValidator] = None


def get_file_validator() -> FileValidator:
    """Retrieve singleton instance of FileValidator."""
    global _file_validator_instance
    if _file_validator_instance is None:
        _file_validator_instance = FileValidator()
    return _file_validator_instance
