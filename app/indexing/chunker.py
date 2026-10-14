"""Structure-aware sliding window text chunker preserving page and document boundaries."""

import re
from typing import List
from app.core.config import get_settings
from app.core.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from app.schemas.ocr import OCRDocumentResult, OCRPage
from app.schemas.retrieval import ChunkMetadata, DocumentChunkItem

settings = get_settings()


class StructureAwareChunker:
    """Chunks documents into semantic passages while preserving page and table context."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, ocr_result: OCRDocumentResult) -> List[DocumentChunkItem]:
        """Split full document result into structured chunks with rich metadata."""
        chunks: List[DocumentChunkItem] = []
        global_chunk_idx = 0

        for page in ocr_result.pages:
            page_chunks = self._chunk_page(
                document_id=ocr_result.document_id,
                filename=ocr_result.filename,
                page=page,
                start_index=global_chunk_idx
            )
            chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        return chunks

    def _chunk_page(
        self,
        document_id: str,
        filename: str,
        page: OCRPage,
        start_index: int
    ) -> List[DocumentChunkItem]:
        """Chunk a single page, treating tables as atomic chunks."""
        page_chunks: List[DocumentChunkItem] = []
        current_idx = start_index

        # 1. Index extracted tables as self-contained atomic chunks
        for tbl in page.tables:
            if tbl.markdown_data.strip():
                chunk_id = f"{document_id}_{current_idx}"
                page_chunks.append(
                    DocumentChunkItem(
                        chunk_id=chunk_id,
                        text=f"### Table on Page {page.page_number}\n{tbl.markdown_data}",
                        metadata=ChunkMetadata(
                            document_id=document_id,
                            filename=filename,
                            page_number=page.page_number,
                            chunk_index=current_idx,
                            char_count=len(tbl.markdown_data),
                            has_table=True,
                            source_type="table"
                        )
                    )
                )
                current_idx += 1

        # 2. Text passages chunking
        raw_text = page.raw_text.strip()
        if not raw_text:
            return page_chunks

        # Split on paragraph boundaries first
        paragraphs = re.split(r"\n\s*\n", raw_text)
        current_text = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_text) + len(para) <= self.chunk_size:
                current_text = f"{current_text}\n\n{para}".strip() if current_text else para
            else:
                if current_text:
                    chunk_id = f"{document_id}_{current_idx}"
                    page_chunks.append(
                        DocumentChunkItem(
                            chunk_id=chunk_id,
                            text=current_text,
                            metadata=ChunkMetadata(
                                document_id=document_id,
                                filename=filename,
                                page_number=page.page_number,
                                chunk_index=current_idx,
                                char_count=len(current_text),
                                has_table=False,
                                source_type="document"
                            )
                        )
                    )
                    current_idx += 1

                    # Retain overlap from end of previous chunk
                    overlap_seed = current_text[-self.chunk_overlap:] if len(current_text) > self.chunk_overlap else ""
                    current_text = f"{overlap_seed}\n{para}".strip()
                else:
                    # Single paragraph exceeds chunk_size, split by sentences
                    current_text = para

        # Flush remaining text
        if current_text:
            chunk_id = f"{document_id}_{current_idx}"
            page_chunks.append(
                DocumentChunkItem(
                    chunk_id=chunk_id,
                    text=current_text,
                    metadata=ChunkMetadata(
                        document_id=document_id,
                        filename=filename,
                        page_number=page.page_number,
                        chunk_index=current_idx,
                        char_count=len(current_text),
                        has_table=False,
                        source_type="document"
                    )
                )
            )

        return page_chunks


_chunker_instance = StructureAwareChunker()


def get_chunker() -> StructureAwareChunker:
    """Retrieve singleton StructureAwareChunker instance."""
    return _chunker_instance
