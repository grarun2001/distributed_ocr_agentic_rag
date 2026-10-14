"""OCR Viewer component displaying extracted text, confidence, engine badges, artifact links, and document previews."""

import io
from typing import Any, Dict, Optional
import requests
import streamlit as st

from frontend.components.table_viewer import render_table_viewer

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None


@st.cache_data(show_spinner=False, ttl=600)
def fetch_document_bytes(raw_url: str) -> Optional[bytes]:
    """Fetch raw file bytes on the server side to bypass browser CORS / MinIO port restrictions."""
    if not raw_url:
        return None
    try:
        resp = requests.get(raw_url, timeout=12)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None


def render_compact_document_image(raw_url: Optional[str], filename: str, width: int = 250) -> None:
    """Render original input document preview with compact, controlled size (not oversized)."""
    st.markdown("#### 🖼️ Original Input Document")
    if not raw_url:
        st.caption("ℹ️ Document storage preview syncing...")
        return

    file_bytes = fetch_document_bytes(raw_url)
    if not file_bytes:
        st.caption("ℹ️ Document storage preview syncing...")
        return

    ext = filename.lower().split(".")[-1] if "." in filename else ""
    is_image = ext in ["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"]
    is_pdf = ext == "pdf"

    if is_image:
        # Compact width display (neat preview, not huge)
        st.image(file_bytes, caption=f"🖼️ {filename} (Preview)", width=width)
        with st.expander("🔍 View Full Size", expanded=False):
            st.image(file_bytes, caption=f"Full Size: {filename}", use_column_width=True)
    elif is_pdf and fitz:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if len(doc) > 0:
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=130)
                st.image(pix.tobytes("png"), caption=f"📄 Page 1 Preview: {filename}", width=width)
                with st.expander("🔍 View Full Size", expanded=False):
                    st.image(pix.tobytes("png"), caption=f"Page 1: {filename}", use_column_width=True)
            else:
                st.info(f"📄 Empty PDF document: {filename}")
        except Exception:
            st.info(f"📄 PDF Document: {filename}")
    else:
        st.info(f"📄 File: {filename}")


def render_ocr_header_metrics(document_data: Optional[Dict[str, Any]] = None) -> None:
    """Render top header, metrics bar, action links, and page summary tabs (Full Width - 1st Image)."""
    if not document_data:
        st.info("Select or process a document to view OCR extractions.")
        return

    st.subheader(f"📄 OCR Extractions: {document_data.get('filename')}")

    # Top Metric Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pages", document_data.get("total_pages", 0))
    with col2:
        st.metric("Indexed Chunks", document_data.get("total_chunks", 0))
    with col3:
        conf = document_data.get("ocr_confidence")
        conf_str = f"{conf:.1f}%" if conf is not None else "N/A"
        st.metric("Average Confidence", conf_str)
    with col4:
        engine = document_data.get("ocr_engine_used", "N/A")
        st.metric("Engine Used", engine)

    # Action links
    raw_url = document_data.get("raw_file_url")
    json_url = document_data.get("ocr_json_url")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        if raw_url:
            st.link_button("📥 Download Raw File", raw_url, use_container_width=True)
    with d_col2:
        if json_url:
            st.link_button("📊 Download Structured OCR JSON", json_url, use_container_width=True)

    # Page breakdown
    pages = document_data.get("pages", [])
    if pages:
        st.markdown("#### Page-by-Page Extraction Summary")
        page_tabs = st.tabs([f"Page {p.get('page_number')}" for p in pages])
        for idx, tab in enumerate(page_tabs):
            p = pages[idx]
            with tab:
                p_col1, p_col2, p_col3 = st.columns(3)
                with p_col1:
                    st.caption(f"**Engine**: `{p.get('ocr_engine')}`")
                with p_col2:
                    p_conf = p.get("confidence")
                    st.caption(f"**Confidence**: `{p_conf:.1f}%`" if p_conf else "**Confidence**: N/A")
                with p_col3:
                    st.caption(f"**Characters**: {p.get('char_count')}")

                warnings = p.get("warnings", [])
                if warnings:
                    st.warning(f"Warnings: {', '.join(warnings)}")


def render_ocr_text_content(document_data: Optional[Dict[str, Any]] = None, height: int = 340) -> None:
    """Render full extracted text content in an interactive viewer (2nd Image text)."""
    if not document_data:
        return

    full_text = document_data.get("ocr_text_preview", "")
    st.markdown("#### 📝 Extracted Text Content")
    if full_text and full_text.strip():
        st.text_area(
            "Document Text",
            value=full_text.strip(),
            height=height,
            disabled=True,
            label_visibility="collapsed"
        )
    else:
        st.info("No extracted text preview available for this document.")


def render_document_ocr_dashboard(document_data: Optional[Dict[str, Any]] = None) -> None:
    """Render complete OCR Extractions Dashboard:
    1. Top (Full Width): Metrics & Page Summary (Image 1)
    2. Middle (Left & Right): Left = Text & Tables (Image 2), Right = Compact Original Document (Image 3)
    """
    if not document_data:
        st.info("No document selected for OCR view.")
        return

    # 1. Top Section: Full Width OCR Extractions Header, Metrics & Page Summary (Image 1)
    render_ocr_header_metrics(document_data)

    st.divider()

    # 2. Middle Section: Left = Extracted Text & Tables (Image 2) | Right = Compact Input Image (Image 3)
    col_text, col_image = st.columns([1.5, 1.0], gap="medium")
    with col_text:
        render_ocr_text_content(document_data, height=340)
        render_table_viewer(document_data)
    with col_image:
        render_compact_document_image(
            document_data.get("raw_file_url"),
            document_data.get("filename", ""),
            width=250
        )


def render_ocr_viewer(document_data: Optional[Dict[str, Any]] = None) -> None:
    """Render complete OCR view (for backward compatibility)."""
    render_document_ocr_dashboard(document_data)
