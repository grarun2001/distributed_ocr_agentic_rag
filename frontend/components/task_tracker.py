"""Task tracker component displaying real-time task lifecycle and progress."""

import time
import streamlit as st
from frontend.api_client import get_api_client

STATUS_COLORS = {
    "PENDING": "#F59E0B",       # Amber
    "PROCESSING": "#3B82F6",    # Blue
    "OCR_COMPLETED": "#06B6D4", # Cyan
    "INDEXING": "#8B5CF6",      # Purple
    "SUCCESS": "#10B981",       # Emerald Green
    "FAILURE": "#EF4444"        # Red
}


def render_task_tracker() -> None:
    """Render current task progress and lifecycle milestone logs."""
    task_id = st.session_state.get("active_task_id")
    if not task_id:
        st.info("ℹ️ No active task in progress. Upload a document to start a processing workflow.")
        return

    st.subheader("⚡ Active Task Monitor")
    client = get_api_client()

    try:
        task_data = client.get_task_status(task_id)
    except Exception as exc:
        st.warning(f"Unable to fetch task status for `{task_id}`: {exc}")
        if st.button("❌ Dismiss Task", key="dismiss_missing_task"):
            st.session_state.pop("active_task_id", None)
            st.rerun()
        return

    status = task_data.get("status", "UNKNOWN")
    progress = float(task_data.get("progress", 0.0))
    stage = task_data.get("current_stage", "Initializing")
    color = STATUS_COLORS.get(status, "#6B7280")

    # Header Card
    st.markdown(
        f"""
        <div style="background-color: #1E293B; border-radius: 8px; padding: 16px; margin-bottom: 16px; border-left: 6px solid {color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 16px; font-weight: 600; color: #F8FAFC;">Task ID: <code>{task_id}</code></span>
                <span style="background-color: {color}; color: #FFFFFF; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold;">
                    {status}
                </span>
            </div>
            <div style="margin-top: 8px; font-size: 14px; color: #94A3B8;">
                Current Stage: <strong style="color: #E2E8F0;">{stage}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Progress bar with stage label
    progress_val = min(max(progress / 100.0, 0.05), 1.0)
    st.progress(progress_val, text=f"⚡ {stage.replace('_', ' ').title()} ({progress:.0f}%)")
    st.caption(f"Pipeline Completion: {progress:.1f}% • Status: {status}")

    if status == "FAILURE":
        st.error(f"❌ Processing failed: {task_data.get('error_message', 'Unknown failure')}")

    # Action buttons: Refresh & Dismiss
    btn_col1, btn_col2, _ = st.columns([1, 1, 2])
    with btn_col1:
        if st.button("🔄 Refresh Status", key="refresh_task_btn"):
            st.rerun()
    with btn_col2:
        if st.button("❌ Dismiss Task", key="dismiss_task_btn"):
            st.session_state.pop("active_task_id", None)
            st.rerun()

    # Milestone History
    logs = task_data.get("logs", [])
    if logs:
        st.markdown("##### 📋 Processing Milestone Logs")
        for log in logs[-5:]:
            st.markdown(
                f"- **[{log.get('stage')}]** ({log.get('progress')}%) - {log.get('message', '')} "
                f"<span style='color: #64748B; font-size: 12px;'>{log.get('created_at', '')}</span>",
                unsafe_allow_html=True
            )

    if status == "SUCCESS":
        doc_id = task_data.get("document_id")
        if doc_id:
            st.session_state["active_document_id"] = doc_id
        # Pop active_task_id so the UI immediately reveals the extractions below
        st.session_state.pop("active_task_id", None)
        st.toast("🎉 Document processed & indexed! Loading extractions...", icon="✅")
        time.sleep(0.3)
        st.rerun()

    elif status in ["PENDING", "PROCESSING", "OCR_COMPLETED", "OCR_EXECUTION", "SAVING_PAGES", "INDEXING"]:
        # Silent automated polling: top progress bar updates smoothly with zero bottom spinner flicker
        time.sleep(1.0)
        st.rerun()

