"""Text normalization and cleanup utilities."""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Perform Unicode NFKC normalization, de-hyphenation, and whitespace consolidation."""
    if not text:
        return ""

    # 1. Unicode NFKC normalization
    normalized = unicodedata.normalize("NFKC", text)

    # 2. Reconnect hyphenated words split across lines (e.g. "distri-\nbuted" -> "distributed")
    dehyphenated = re.sub(r"(\w+)-\n\s*(\w+)", r"\1\2", normalized)

    # 3. Remove non-printable control characters (retain newlines and tabs)
    clean_chars = "".join(
        ch for ch in dehyphenated
        if ch in "\n\r\t" or unicodedata.category(ch)[0] != "C"
    )

    # 4. Standardize line endings and collapse excessive blank lines (> 2 into 2)
    clean_newlines = re.sub(r"\r\n|\r", "\n", clean_chars)
    collapsed_newlines = re.sub(r"\n{3,}", "\n\n", clean_newlines)

    # 5. Consolidate multiple inline spaces/tabs into a single space
    clean_spaces = re.sub(r"[ \t]{2,}", " ", collapsed_newlines)

    return clean_spaces.strip()
