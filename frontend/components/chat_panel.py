import re
from typing import Optional
import streamlit as st

from frontend.api_client import get_api_client
from frontend.components.source_viewer import render_source_viewer

SOURCE_COLORS = {
    "DOCUMENTS": "#3B82F6",             # Blue
    "WEB": "#10B981",                   # Emerald
    "HYBRID": "#8B5CF6",                # Purple
    "INSUFFICIENT_EVIDENCE": "#F59E0B"  # Amber
}


def format_markdown_for_display(content: str) -> str:
    """Ensure bullet points and list items are separated by blank lines so markdown renders one-by-one."""
    if not content:
        return ""
    text = str(content).replace("\r\n", "\n").replace("\r", "\n")
    # Convert unicode bullet points (•) to markdown dash list items (- )
    text = re.sub(r"(?<=\S)\s*[•]\s*", "\n\n- ", text)
    text = re.sub(r"^[ \t]*[•][ \t]*", "- ", text, flags=re.MULTILINE)
    # Ensure double newline before list items so they never collapse into the same paragraph
    text = re.sub(r"(?<!\n)\n(?=- )", "\n\n", text)
    return text


def render_chat_stream(doc_filter: Optional[str], allow_web: bool) -> None:
    """Render chat messages and the prompt input box."""
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Hello! I am your Document AI Assistant. Ask me anything about your uploaded files.",
                "source_type": "SYSTEM",
                "confidence_score": 1.0,
                "thought_steps": [],
                "document_citations": [],
                "web_citations": [],
                "latency_ms": 0.0
            }
        ]

    # Render message history
    for msg in st.session_state["messages"]:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(format_markdown_for_display(msg["content"]))

            if role == "assistant" and msg.get("source_type") != "SYSTEM":
                source_type = msg.get("source_type", "DOCUMENTS")
                badge_color = SOURCE_COLORS.get(source_type, "#6B7280")
                conf = msg.get("confidence_score", 0.0)

                # Clean minimalist badges
                st.markdown(
                    f"""
                    <div style="display: flex; gap: 8px; align-items: center; margin-top: 4px; margin-bottom: 6px;">
                        <span style="background-color: {badge_color}; color: white; padding: 2px 8px; border-radius: 8px; font-size: 11px; font-weight: 600;">
                            {source_type}
                        </span>
                        <span style="color: #64748B; font-size: 11px;">
                            Confidence: <b>{conf * 100:.0f}%</b>
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Thought Process Expander (collapsed by default)
                thought_steps = msg.get("thought_steps", [])
                if thought_steps:
                    with st.expander("🧠 Agent Reasoning", expanded=False):
                        for step in thought_steps:
                            agent_name = step.get("agent_name", "Agent")
                            action = step.get("action", "ACTION")
                            rationale = step.get("rationale", "")
                            st.markdown(f"- **{agent_name}** (`{action}`): {rationale}")

                # Citations
                doc_cites = msg.get("document_citations", [])
                web_cites = msg.get("web_citations", [])
                if doc_cites or web_cites:
                    render_source_viewer(doc_cites, web_cites)

    # Prompt Input Box
    placeholder = "Ask a question about this document..." if doc_filter else "Ask a question across all documents..."
    if prompt := st.chat_input(placeholder):
        # 1. Immediately append and display user message
        st.session_state["messages"].append({
            "role": "user",
            "content": prompt,
            "source_type": "USER",
            "confidence_score": 1.0,
            "thought_steps": [],
            "document_citations": [],
            "web_citations": [],
            "latency_ms": 0.0
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Query Agentic RAG graph with spinner
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document knowledge & generating answer..."):
                client = get_api_client()
                try:
                    res = client.ask_question(
                        question=prompt,
                        document_id=doc_filter,
                        top_k=4,
                        allow_web_search=allow_web
                    )
                    answer = res.get("answer", "No answer generated.")
                    st.markdown(format_markdown_for_display(answer))

                    source_type = res.get("source_type", "DOCUMENTS")
                    badge_color = SOURCE_COLORS.get(source_type, "#6B7280")
                    conf = res.get("confidence_score", 0.0)
                    lat = res.get("latency_ms", 0.0)

                    st.markdown(
                        f"""
                        <div style="display: flex; gap: 8px; align-items: center; margin-top: 4px; margin-bottom: 6px;">
                            <span style="background-color: {badge_color}; color: white; padding: 2px 8px; border-radius: 8px; font-size: 11px; font-weight: 600;">
                                {source_type}
                            </span>
                            <span style="color: #64748B; font-size: 11px;">
                                Confidence: <b>{conf * 100:.0f}%</b>
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    thought_steps = res.get("thought_steps", [])
                    if thought_steps:
                        with st.expander("🧠 Agent Reasoning", expanded=False):
                            for step in thought_steps:
                                agent_name = step.get("agent_name", "Agent")
                                action = step.get("action", "ACTION")
                                rationale = step.get("rationale", "")
                                st.markdown(f"- **{agent_name}** (`{action}`): {rationale}")

                    doc_cites = res.get("document_citations", [])
                    web_cites = res.get("web_citations", [])
                    if doc_cites or web_cites:
                        render_source_viewer(doc_cites, web_cites)

                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "source_type": source_type,
                        "confidence_score": conf,
                        "thought_steps": thought_steps,
                        "document_citations": doc_cites,
                        "web_citations": web_cites,
                        "latency_ms": lat
                    })

                except Exception as exc:
                    err_msg = f"❌ Error: {exc}"
                    st.error(err_msg)
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": err_msg,
                        "source_type": "INSUFFICIENT_EVIDENCE",
                        "confidence_score": 0.0,
                        "thought_steps": [],
                        "document_citations": [],
                        "web_citations": [],
                        "latency_ms": 0.0
                    })


def render_chat_panel(selected_document_id: Optional[str] = None, doc_data: Optional[dict] = None) -> None:
    """Render pure Agentic RAG Q&A Assistant: Ask questions and view answers with citations."""
    client = get_api_client()
    if selected_document_id and not doc_data:
        try:
            doc_data = client.get_document(selected_document_id)
        except Exception:
            doc_data = None

    # Control & Scope Bar
    head_col1, head_col2, head_col3 = st.columns([3, 1, 1])
    with head_col1:
        if doc_data:
            st.subheader(f"💬 Ask Questions About `{doc_data.get('filename')}`")
            st.caption(f"Context is grounded on document extractions ({doc_data.get('total_chunks', 0)} chunks indexed).")
        else:
            st.subheader("💬 Ask Questions Across All Documents")
            st.caption("Context is grounded across all indexed documents in the repository.")
    with head_col2:
        allow_web = st.toggle("🌐 Web Search", value=False, help="Search the web if document evidence is insufficient")
    with head_col3:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    # Chat Stream (Messages + Input box at bottom)
    render_chat_stream(doc_filter=selected_document_id, allow_web=allow_web)
