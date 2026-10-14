"""Unit tests for structure-aware passage and table chunking."""

from app.indexing.chunker import StructureAwareChunker
from app.schemas.ocr import OCRBlock, OCRBoundingBox, OCRDocumentResult, OCRPage, OCRTable


def test_chunker_page_and_table_preservation():
    """Verify chunker isolates tables as individual chunks and tracks page numbers."""
    chunker = StructureAwareChunker(chunk_size=200, chunk_overlap=20)

    table = OCRTable(
        table_index=0,
        num_rows=2,
        num_cols=2,
        headers=["Item", "Cost"],
        rows=[["Server", "$1000"]],
        markdown_data="| Item | Cost |\n| --- | --- |\n| Server | $1000 |",
        confidence=100.0
    )

    page1 = OCRPage(
        page_number=1,
        ocr_engine="NATIVE",
        confidence=100.0,
        char_count=150,
        table_count=1,
        has_selectable_text=True,
        raw_text="Introduction to distributed systems. Reliability is fundamental.",
        blocks=[],
        tables=[table],
        warnings=[]
    )

    page2 = OCRPage(
        page_number=2,
        ocr_engine="NATIVE",
        confidence=100.0,
        char_count=100,
        table_count=0,
        has_selectable_text=True,
        raw_text="Conclusion and next steps for deployment.",
        blocks=[],
        tables=[],
        warnings=[]
    )

    doc_result = OCRDocumentResult(
        document_id="doc_test_123",
        filename="test_report.pdf",
        category="PDF_DIGITAL",
        total_pages=2,
        average_confidence=100.0,
        ocr_engine_used="NATIVE",
        pages=[page1, page2],
        full_text="Combined text content",
        warnings=[]
    )

    chunks = chunker.chunk_document(doc_result)

    assert len(chunks) >= 3  # Table chunk + Page 1 text chunk + Page 2 text chunk

    # Verify table chunk
    table_chunk = next(c for c in chunks if c.metadata.has_table)
    assert table_chunk.metadata.page_number == 1
    assert "Table on Page 1" in table_chunk.text
    assert table_chunk.metadata.source_type == "table"

    # Verify regular text chunks
    p2_chunk = next(c for c in chunks if c.metadata.page_number == 2)
    assert "Conclusion and next steps" in p2_chunk.text
    assert p2_chunk.metadata.document_id == "doc_test_123"
    assert p2_chunk.metadata.filename == "test_report.pdf"
