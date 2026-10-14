"""Preserves headings, lists, tables, and paragraphs during text extraction."""

import re
from typing import List


def preserve_document_structure(text: str) -> str:
    """Format detected structural components (headings, bullets) into clean Markdown without corrupting tables or prices."""
    if not text:
        return ""

    lines = text.split("\n")
    processed_lines: List[str] = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            processed_lines.append("")
            continue

        # Toggle code block state
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            processed_lines.append(stripped)
            continue

        if in_code_block:
            processed_lines.append(line)
            continue

        # 1. Protect Markdown tables: do not alter table rows
        if stripped.startswith("|") and stripped.endswith("|"):
            processed_lines.append(stripped)
            continue

        # 2. Protect existing Markdown headings and blockquotes
        if stripped.startswith(("#", ">")):
            processed_lines.append(stripped)
            continue

        # 3. Detect formal numbered section headings (e.g. "SECTION 1: OVERVIEW", "1.1 Introduction")
        if re.match(r"^(SECTION|CHAPTER|ARTICLE|APPENDIX)\s+[A-Z0-9\.]+", stripped, re.IGNORECASE):
            processed_lines.append(f"\n### {stripped}\n")
            continue
        elif re.match(r"^\d+(\.\d+)+\s+[A-Za-z]", stripped):
            processed_lines.append(f"\n### {stripped}\n")
            continue

        # 4. Protect key-value pairs and prices (e.g. "Total: $50.00", "Date: 2024-01-01")
        if ":" in stripped and not stripped.endswith(":"):
            # Likely a key-value pair, preserve verbatim
            processed_lines.append(stripped)
            continue

        # 5. Detect all-caps title lines (pure words only, 4 to 50 chars, no currency/symbols)
        elif (
            re.match(r"^[A-Z][A-Z\s]{3,50}$", stripped)
            and len(stripped.split()) <= 6
            and not any(c in stripped for c in "$€₹£%=|")
        ):
            processed_lines.append(f"\n## {stripped}\n")

        # 6. Standardize bullet points
        elif re.match(r"^[\*\•\-\–]\s+", stripped):
            cleaned_bullet = re.sub(r"^[\*\•\-\–]\s+", "- ", stripped)
            processed_lines.append(cleaned_bullet)

        else:
            processed_lines.append(stripped)

    return "\n".join(processed_lines).strip()

