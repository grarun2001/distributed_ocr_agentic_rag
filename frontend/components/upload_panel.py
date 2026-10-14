"""Upload panel component handling file drop, validation display, and async ingestion."""

import streamlit as st
from frontend.api_client import get_api_client


def render_upload_panel() -> None:
    """Render the file upload dropzone and ingestion controls."""
    st.subheader("📤 Document Ingestion")
    st.markdown(
        "Upload any business document, spreadsheet, presentation, or scanned image. "
        "The distributed pipeline performs cryptographic SHA-256 fingerprinting, layout analysis, "
        "OCR, table extraction, and dual vector + sparse indexing."
    )

    # Format badges
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.caption("📄 **PDF & Scans**: PDF, TIFF, PNG, JPG")
    with col2:
        st.caption("📝 **Documents**: DOCX, RTF, TXT")
    with col3:
        st.caption("📊 **Spreadsheets**: XLSX, CSV")
    with col4:
        st.caption("📽️ **Other**: PPTX, HTML")

    uploaded_file = st.file_uploader(
        "Choose a file to ingest (Max 50MB)",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "docx", "xlsx", "pptx", "rtf", "html", "txt", "csv"],
        help="Upload digital or scanned documents for distributed processing"
    )

    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"📁 Selected: **{uploaded_file.name}** ({file_size_mb:.2f} MB)")

        if st.button("🚀 Ingest & Process Document", use_container_width=True, type="primary"):
            client = get_api_client()
            with st.spinner("Uploading to secure MinIO vault & dispatching Celery task..."):
                try:
                    res = client.upload_file(
                        file_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                        content_type=uploaded_file.type or "application/octet-stream"
                    )
                    st.success(f"✅ Ingestion accepted! Task ID: `{res['task_id']}`")
                    st.session_state["active_task_id"] = res["task_id"]
                    st.session_state["active_document_id"] = res["document_id"]
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ Upload failed: {exc}")
