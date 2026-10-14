"""Citation builder and verifier separating document evidence and web sources."""

import re
from typing import Dict, List, Tuple
from app.agents.guardrails import validate_citations_against_grounding
from app.schemas.answer import DocumentCitation, WebCitation
from app.schemas.retrieval import RetrievalHit


class CitationBuilder:
    """Builds and validates verifiable citations for documents and web sources."""

    def extract_and_validate_citations(
        self,
        answer_text: str,
        retrieved_chunks: List[RetrievalHit],
        web_results: List[Dict[str, str]]
    ) -> Tuple[List[DocumentCitation], List[WebCitation]]:
        """
        Extract citations from answer text or construct them from the top ground truth evidence.
        Guarantees that no fabricated citation passes through.
        """
        doc_citations: List[DocumentCitation] = []
        web_citations: List[WebCitation] = []

        # 1. Regex search for explicit [Doc: ...] tags
        doc_pattern = r"\[Doc:\s*([^,\]]+)(?:,\s*Page:\s*(\d+))?\]"
        matches = re.findall(doc_pattern, answer_text, re.IGNORECASE)

        chunk_lookup = {c.filename.lower(): c for c in retrieved_chunks}

        for filename_match, page_match in matches:
            clean_fn = filename_match.strip().lower()
            if clean_fn in chunk_lookup:
                matched_chunk = chunk_lookup[clean_fn]
                doc_citations.append(
                    DocumentCitation(
                        document_id=matched_chunk.document_id,
                        filename=matched_chunk.filename,
                        page_number=int(page_match) if page_match else matched_chunk.page_number,
                        chunk_index=matched_chunk.chunk_index,
                        snippet=matched_chunk.text[:200]
                    )
                )

        # 2. Regex search for explicit [Web: ...] tags
        web_pattern = r"\[Web:\s*([^\]]+)\]"
        web_matches = re.findall(web_pattern, answer_text, re.IGNORECASE)

        url_lookup = {r.get("url", "").lower(): r for r in web_results if r.get("url")}
        title_lookup = {r.get("title", "").lower(): r for r in web_results if r.get("title")}

        for web_match in web_matches:
            target = web_match.strip().lower()
            matched_res = None
            if target in url_lookup:
                matched_res = url_lookup[target]
            elif target in title_lookup:
                matched_res = title_lookup[target]
            else:
                # Substring match
                for url, res in url_lookup.items():
                    if target in url or target in res.get("title", "").lower():
                        matched_res = res
                        break

            if matched_res:
                web_citations.append(
                    WebCitation(
                        title=matched_res.get("title", "Web Source"),
                        url=matched_res.get("url", ""),
                        snippet=matched_res.get("body", "")[:200]
                    )
                )

        # 3. Fallback: If no explicit inline citation tags were found in text:
        # Only attach document chunks if they have genuine relevance (score >= 0.25),
        # not if the answer came from web or the text states lack of document information.
        no_doc_info = "no information" in answer_text.lower() or "insufficient evidence" in answer_text.lower()
        if not doc_citations and retrieved_chunks and not web_citations and not no_doc_info:
            for top_c in retrieved_chunks[:2]:
                if getattr(top_c, "score", 0.0) >= 0.25:
                    doc_citations.append(
                        DocumentCitation(
                            document_id=top_c.document_id,
                            filename=top_c.filename,
                            page_number=top_c.page_number,
                            chunk_index=top_c.chunk_index,
                            snippet=top_c.text[:200]
                        )
                    )

        if not web_citations and web_results:
            for top_w in web_results[:2]:
                web_citations.append(
                    WebCitation(
                        title=top_w.get("title", "Web Source"),
                        url=top_w.get("url", ""),
                        snippet=top_w.get("body", "")[:200]
                    )
                )

        # 4. Strict guardrail against fabricated citations
        return validate_citations_against_grounding(
            doc_citations,
            web_citations,
            retrieved_chunks,
            web_results
        )


_citation_builder_instance = CitationBuilder()


def get_citation_builder() -> CitationBuilder:
    """Retrieve singleton instance of CitationBuilder."""
    return _citation_builder_instance
