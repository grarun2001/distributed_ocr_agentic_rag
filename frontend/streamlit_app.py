"""Streamlit Frontend Application for Distributed OCR & Agentic RAG System."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from frontend.api_client import get_api_client
from frontend.components.chat_panel import render_chat_panel
from frontend.components.ocr_viewer import render_ocr_viewer
from frontend.components.table_viewer import render_table_viewer
from frontend.components.task_tracker import render_task_tracker
from frontend.components.upload_panel import render_upload_panel

# Configure page layout and title
st.set_page_config(
    page_title="Distributed OCR & Agentic RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark UI aesthetics
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Card styling */
    .stCard {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #334155;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 700;
        color: #38BDF8;
    }

    /* Sidebar navigation styling */
    div[data-testid="stRadio"] > div {
        gap: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar() -> Optional[str]:
    """Render sidebar containing system status, active document workspace selector, and management actions."""
    with st.sidebar:
        st.markdown("## ⚡ **OCR Agentic RAG**")
        st.caption("Distributed Document Intelligence & Multi-Agent Assistant")

        st.divider()

        # Health status monitor
        client = get_api_client()
        try:
            health = client.check_health()
            status_overall = health.get("status", "UNREACHABLE")
        except Exception:
            health = {}
            status_overall = "UNREACHABLE"

        if status_overall == "HEALTHY":
            st.success("🟢 System Cluster: **ONLINE**")
        elif status_overall == "DEGRADED":
            st.warning("🟡 System Cluster: **DEGRADED**")
        else:
            st.error("🔴 System Cluster: **OFFLINE**")

        deps = health.get("dependencies", {})
        if deps:
            with st.expander("🔌 Service Infrastructure", expanded=False):
                for dep, state in deps.items():
                    icon = "✅" if state in ["OK", "HEALTHY", "connected"] else "⚠️"
                    st.markdown(f"- **{dep.capitalize()}**: {icon} `{state}`")

        st.divider()

        # Active Document Selector
        st.markdown("### 📁 Document Workspace")
        docs = []
        total_docs = 0
        try:
            docs_res = client.list_documents(limit=100)
            docs = docs_res.get("items", [])
            total_docs = docs_res.get("total", len(docs))
            if docs:
                st.session_state["_cached_docs"] = docs
                st.session_state["_cached_total_docs"] = total_docs
        except Exception:
            if "_cached_docs" in st.session_state:
                docs = st.session_state["_cached_docs"]
                total_docs = st.session_state.get("_cached_total_docs", len(docs))

        doc_options = {}
        used_labels = set()
        for d in docs:
            name = d.get("filename", "Untitled")
            label = f"📄 {name}"
            if label in used_labels:
                label = f"📄 {name} ({d['id'][:8]})"
            used_labels.add(label)
            doc_options[label] = d["id"]

        if total_docs > 1:
            doc_options["🌐 Global (All Documents)"] = None

        current_active = st.session_state.get("active_document_id")
        selected_index = 0
        if current_active:
            for idx, (label, doc_id) in enumerate(doc_options.items()):
                if doc_id == current_active:
                    selected_index = idx
                    break
        elif doc_options:
            # If nothing selected yet, prefer selecting Leave_Attendance_Policy.pdf or the first document
            for idx, (label, doc_id) in enumerate(doc_options.items()):
                if "Leave_Attendance_Policy" in label:
                    selected_index = idx
                    break

        if doc_options:
            selected_label = st.selectbox(
                "Choose Active Document",
                options=list(doc_options.keys()),
                index=selected_index,
                help="Select which document to inspect extractions and chat with"
            )
            selected_id = doc_options[selected_label]
            st.session_state["active_document_id"] = selected_id
            st.session_state["selected_scope_label"] = selected_label
        else:
            selected_id = None
            st.session_state["active_document_id"] = None
            st.session_state["selected_scope_label"] = None
            st.caption("ℹ️ No documents uploaded yet.")

        # Delete Actions
        if selected_id:
            if st.button("🗑️ Delete This Document", use_container_width=True):
                try:
                    client.delete_document(selected_id)
                    st.toast(f"Deleted document successfully!", icon="🗑️")
                    st.session_state.pop("active_document_id", None)
                    st.cache_data.clear()
                    st.rerun()
                except Exception as del_err:
                    st.error(f"Failed to delete: {del_err}")

        if total_docs > 0:
            with st.expander("⚙️ Repository Management", expanded=False):
                st.caption("Delete all documents to start completely fresh:")
                if st.button("🧹 Clear All Documents", use_container_width=True):
                    for d in docs:
                        try:
                            client.delete_document(d["id"])
                        except Exception:
                            pass
                    st.toast("All documents cleared!", icon="🧹")
                    st.session_state.pop("active_document_id", None)
                    st.cache_data.clear()
                    st.rerun()

        st.caption(f"Total Ingested Documents: **{total_docs}**")

        return selected_id


def render_document_library() -> None:
    """Render full document repository management view."""
    st.subheader("📚 Ingested Document Repository")
    client = get_api_client()

    try:
        docs_res = client.list_documents(limit=50)
        items = docs_res.get("items", [])
    except Exception as exc:
        st.error(f"Failed to load document library: {exc}")
        return

    if not items:
        st.info("No documents uploaded yet. Use the Upload view to add files.")
        return

    # Document table summary
    table_data = []
    for d in items:
        table_data.append({
            "Document ID": d["id"],
            "Filename": d["filename"],
            "Category": d["category"],
            "Status": d["status"],
            "Pages": d["total_pages"],
            "Chunks": d["total_chunks"],
            "Engine": d.get("ocr_engine_used", "N/A"),
            "Confidence": f"{d['ocr_confidence']:.1f}%" if d.get("ocr_confidence") else "N/A",
            "Created": d["created_at"][:19]
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)

    # Actions for selected document
    st.markdown("#### Document Actions")
    sel_id = st.selectbox(
        "Choose document to inspect or manage",
        options=[d["id"] for d in items],
        format_func=lambda x: next(d["filename"] for d in items if d["id"] == x)
    )

    if sel_id:
        try:
            doc_detail = client.get_document(sel_id)
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if doc_detail.get("raw_file_url"):
                    st.link_button("📥 Raw Document", doc_detail["raw_file_url"], use_container_width=True)
            with c2:
                if doc_detail.get("ocr_json_url"):
                    st.link_button("📊 Full OCR JSON", doc_detail["ocr_json_url"], use_container_width=True)
            with c3:
                if st.button("🗑️ Delete Document", type="secondary", use_container_width=True):
                    with st.spinner("Cascading deletion across Postgres, MinIO, and Milvus..."):
                        client.delete_document(sel_id)
                        st.success("Document deleted!")
                        st.rerun()

            # Render OCR and Table details
            st.divider()
            render_ocr_viewer(doc_detail)
            render_table_viewer(doc_detail)
        except Exception as exc:
            st.warning(f"Could not load details for document `{sel_id}`: {exc}")


def render_search_explorer() -> None:
    """Render interactive hybrid search and reranking explorer."""
    st.subheader("🔍 Hybrid Search & Reranking Explorer")
    st.markdown(
        "Directly test the retrieval engine. Compare Milvus dense vector search, persistent BM25 keyword matching, "
        "Reciprocal Rank Fusion (RRF), and Cross-Encoder neural reranking."
    )

    query = st.text_input("Enter search query or keyword phrase:", placeholder="e.g. quarterly revenue or wire transfer")
    col1, col2, col3 = st.columns(3)
    with col1:
        mode = st.selectbox(
            "Retrieval Strategy",
            options=["HYBRID_RERANK", "HYBRID_RRF", "VECTOR_ONLY", "BM25_ONLY"],
            help="Select fusion or individual retrieval mode"
        )
    with col2:
        top_k = st.slider("Result Count (Top K)", min_value=1, max_value=10, value=5)
    with col3:
        filter_doc = st.text_input("Document ID Filter (Optional)", placeholder="Leave blank for global search")

    if st.button("🔎 Run Retrieval", type="primary") and query:
        client = get_api_client()
        with st.spinner("Searching Milvus and persistent BM25..."):
            try:
                res = client.hybrid_search(
                    query=query,
                    top_k=top_k,
                    document_id=filter_doc.strip() if filter_doc.strip() else None,
                    mode=mode
                )
                hits = res.get("hits", [])
                st.success(f"Retrieved **{len(hits)}** ranked hit(s) using **{mode}**")

                for idx, hit in enumerate(hits, start=1):
                    with st.expander(f"Hit #{idx}: {hit['filename']} (Page {hit['page_number']}) - Score: {hit['score']:.4f} [{hit['retrieval_source']}]", expanded=(idx <= 2)):
                        st.markdown(f"**Chunk ID**: `{hit['chunk_id']}`")
                        st.markdown(f"**Relevance Score**: `{hit['score']:.4f}` | **Source**: `{hit['retrieval_source']}`")
                        st.text_area("Passage Text", hit["text"], height=120, disabled=True, key=f"hit_{idx}")
            except Exception as exc:
                st.error(f"Search failed: {exc}")


def main() -> None:
    """Unified Single-Page Workspace:
    1. Top: Upload Document & Live Processing Task Tracker (collapsible expander)
    2. Middle: OCR Extractions Dashboard (Full metrics at top, Text & Tables on left, Compact Image on right)
    3. Bottom: Agentic RAG Q&A Assistant directly below OCR extractions!
    """
    selected_id = render_sidebar()
    client = get_api_client()

    has_active_task = bool(st.session_state.get("active_task_id"))
    selected_scope_label = st.session_state.get("selected_scope_label", "")

    # 1. Top Section: Document Upload Dropzone & Task Monitor
    with st.expander("📤 Upload New Document / Processing Tracker", expanded=has_active_task or not bool(selected_id)):
        _, center_col, _ = st.columns([0.05, 0.9, 0.05])
        with center_col:
            render_upload_panel()
            st.divider()
            render_task_tracker()

    # 2. Middle & Bottom Sections: Unified OCR Extractions & Q&A Assistant
    doc_data = None
    if selected_id:
        try:
            doc_data = client.get_document(selected_id)
        except Exception as exc:
            st.warning(f"Unable to load document details for `{selected_id}`: {exc}")

    if doc_data:
        # OCR Extractions Dashboard: Full-width Metrics at top, Text & Tables on Left, Compact Preview on Right
        st.divider()
        render_ocr_viewer(doc_data)

        # Directly below OCR: Question Asking Section!
        st.divider()
        render_chat_panel(selected_document_id=selected_id, doc_data=doc_data)

    elif selected_scope_label == "🌐 Global (All Documents)":
        st.divider()
        render_chat_panel(selected_document_id=None, doc_data=None)

    elif not selected_id and not has_active_task:
        st.info("👋 Upload a document above or select an existing document from the sidebar to inspect extractions and ask questions!")


if __name__ == "__main__":
    main()
