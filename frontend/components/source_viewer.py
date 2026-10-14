import re
from typing import Any, Dict, List
import streamlit as st


def render_source_viewer(
    document_citations: List[Dict[str, Any]],
    web_citations: List[Dict[str, Any]]
) -> None:
    """Render verified document and web citations in clean, compact, proportional cards."""
    if not document_citations and not web_citations:
        st.caption("No external sources cited for this response.")
        return

    st.markdown("##### 🔍 Grounded Evidence & Verified Citations")

    # Document Citations
    if document_citations:
        st.markdown("**📄 Ingested Document Sources:**")
        for idx, doc_cite in enumerate(document_citations, start=1):
            fn = doc_cite.get("filename", "Unknown Document")
            page = doc_cite.get("page_number", 1)
            raw_snip = doc_cite.get("snippet", "")
            clean_snip = re.sub(r"\s+", " ", str(raw_snip or "")).strip()
            if len(clean_snip) > 180:
                clean_snip = clean_snip[:180].rstrip() + "..."

            st.markdown(
                f"""
                <div style="background-color: #0F172A; border-left: 4px solid #6366F1; border-radius: 6px; padding: 8px 12px; margin-bottom: 6px;">
                    <div style="font-weight: 600; font-size: 12px; color: #818CF8;">
                        [Doc #{idx}] {fn} (Page {page})
                    </div>
                    <div style="font-size: 12px; color: #CBD5E1; margin-top: 3px; font-style: italic; line-height: 1.4;">
                        "{clean_snip}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Web Citations
    if web_citations:
        st.markdown("**🌐 External Live Web Sources:**")
        for idx, web_cite in enumerate(web_citations, start=1):
            title = web_cite.get("title", "Web Source")
            url = web_cite.get("url", "#")
            raw_snip = web_cite.get("snippet", "")
            clean_snip = re.sub(r"\s+", " ", str(raw_snip or "")).strip()
            if len(clean_snip) > 180:
                clean_snip = clean_snip[:180].rstrip() + "..."

            st.markdown(
                f"""
                <div style="background-color: #0F172A; border-left: 4px solid #10B981; border-radius: 6px; padding: 8px 12px; margin-bottom: 6px;">
                    <div style="font-weight: 600; font-size: 12px; color: #34D399;">
                        [Web #{idx}] <a href="{url}" target="_blank" style="color: #38BDF8; text-decoration: none;">{title}</a>
                    </div>
                    <div style="font-size: 12px; color: #CBD5E1; margin-top: 3px; font-style: italic; line-height: 1.4;">
                        "{clean_snip}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

