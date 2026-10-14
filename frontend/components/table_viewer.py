"""Table viewer component rendering extracted tabular data and CSV export."""

import io
from typing import Any, Dict, List, Optional
import pandas as pd
import requests
import streamlit as st


def render_table_viewer(document_data: Optional[Dict[str, Any]] = None) -> None:
    """Extract and render tables found in the document artifact."""
    if not document_data:
        st.info("Select a document to inspect extracted tables.")
        return

    st.subheader(f"📊 Extracted Tables: {document_data.get('filename')}")

    ocr_json_url = document_data.get("ocr_json_url")
    tables: List[Dict[str, Any]] = []

    if ocr_json_url:
        try:
            resp = requests.get(ocr_json_url, timeout=10)
            if resp.status_code == 200:
                doc_json = resp.json()
                for page in doc_json.get("pages", []):
                    for tbl in page.get("tables", []):
                        tbl["page_number"] = page.get("page_number", 1)
                        tables.append(tbl)
        except Exception:
            pass

    if not tables:
        st.info("ℹ️ No structured tables detected in this document.")
        return

    st.success(f"Detected **{len(tables)}** structured table(s) across document pages.")

    for idx, tbl in enumerate(tables, start=1):
        with st.expander(f"Table #{idx} (Page {tbl.get('page_number', 1)})", expanded=True):
            headers = tbl.get("headers", [])
            rows = tbl.get("rows", [])

            if rows:
                if headers and len(headers) == len(rows[0]):
                    df = pd.DataFrame(rows, columns=headers)
                else:
                    df = pd.DataFrame(rows)

                # Render interactive DataFrame
                st.dataframe(df, use_container_width=True)

                # CSV download
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label=f"📥 Download Table #{idx} as CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"{document_data.get('filename')}_table_{idx}.csv",
                    mime="text/csv",
                    key=f"csv_tbl_{document_data.get('id')}_{idx}"
                )
            elif tbl.get("markdown_data"):
                st.markdown(tbl.get("markdown_data"))
