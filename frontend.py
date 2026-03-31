"""Streamlit frontend for NerdFarm Document Understanding System."""

import asyncio
import httpx
import logging
import streamlit as st
from datetime import datetime

logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="NerdFarm",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS
st.markdown("""
<style>
    .stMetric label { font-size: 0.85rem; }
    .stMetric value { font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "uploaded_docs" not in st.session_state or st.session_state.uploaded_docs is None:
        st.session_state.uploaded_docs = []
        # Load persisted documents from backend
        try:
            response = asyncio.run(api_request("GET", "/api/documents"))
            if response.status_code == 200:
                docs = response.json()
                st.session_state.uploaded_docs = [
                    {
                        "id": doc["document_id"],
                        "filename": doc["filename"],
                        "pages": doc["total_pages"],
                        "chunks": doc.get("chunks_created", 0),
                        "confidence": doc.get("avg_confidence", 0),
                    }
                    for doc in docs
                ]
        except Exception:
            pass  # Backend may not be running yet
    if "current_draft" not in st.session_state:
        st.session_state.current_draft = None
    if "edit_content" not in st.session_state:
        st.session_state.edit_content = ""
    if "edit_history" not in st.session_state:
        st.session_state.edit_history = []


async def api_request(method: str, endpoint: str, **kwargs):
    async with httpx.AsyncClient(timeout=60.0) as client:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            return await client.get(url, **kwargs)
        elif method == "POST":
            return await client.post(url, **kwargs)
        elif method == "DELETE":
            return await client.delete(url, **kwargs)



def main():
    init_session_state()

    with st.sidebar:
        st.markdown("### 📄 NerdFarm")
        st.caption("Document Understanding System")
        st.divider()

        try:
            response = asyncio.run(api_request("GET", "/health"))
            st.success("✅ Connected" if response.status_code == 200 else "❌ Error")
        except Exception:
            st.error("❌ Backend offline")

        st.divider()
        page = st.radio(
            "Navigate",
            ["🏠 Home", "📤 Upload", "📝 Generate", "✏️ Edit", "📊 Learning"],
            label_visibility="collapsed",
        )
        st.divider()
        st.metric("Documents", len(st.session_state.uploaded_docs))
        st.metric("Edits", len(st.session_state.edit_history))

    if "Home" in page:
        show_home_page()
    elif "Upload" in page:
        show_upload_page()
    elif "Generate" in page:
        show_generate_page()
    elif "Edit" in page:
        show_edit_page()
    elif "Learning" in page:
        show_learning_page()


def show_home_page():
    st.title("🏠 NerdFarm")
    st.markdown("Document Understanding & Grounded Drafting System")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📤 Process Documents")
        st.markdown("Upload PDFs, images, or text files for OCR and indexing.")
    with col2:
        st.markdown("#### 📝 Generate Drafts")
        st.markdown("Create grounded drafts with citations from your documents.")
    with col3:
        st.markdown("#### 📚 Learn from Edits")
        st.markdown("System improves from your corrections over time.")

    st.divider()
    st.markdown("**Workflow:** Upload → Generate → Edit → System Learns")


def show_upload_page():
    st.title("📤 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose files (PDF, images, or text)",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            st.caption(f"📄 {f.name} ({f.size / 1024:.1f} KB)")

        if st.button("Process All", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            success_count = 0
            error_count = 0

            for i, uploaded_file in enumerate(uploaded_files):
                status_text.markdown(f"Processing: **{uploaded_file.name}** ({i+1}/{len(uploaded_files)})")
                progress_bar.progress((i) / len(uploaded_files))

                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = asyncio.run(api_request("POST", "/api/documents/upload", files=files))

                    if response.status_code == 200:
                        result = response.json()
                        success_count += 1

                        # Reload documents from backend to get accurate list
                        st.session_state.uploaded_docs = None  # Force reload
                        init_session_state()
                    else:
                        error_count += 1
                        st.warning(f"❌ {uploaded_file.name}: {response.text[:100]}")
                except Exception as e:
                    error_count += 1
                    st.warning(f"❌ {uploaded_file.name}: {str(e)[:100]}")

            progress_bar.progress(1.0)
            status_text.empty()

            if success_count > 0:
                st.success(f"✅ Successfully processed {success_count} document(s)")
            if error_count > 0:
                st.error(f"❌ Failed to process {error_count} document(s)")

    if st.session_state.uploaded_docs:
        st.divider()
        st.markdown(f"#### 📁 Uploaded Documents ({len(st.session_state.uploaded_docs)})")
        for i, doc in enumerate(st.session_state.uploaded_docs):
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 0.5])
            col1.markdown(f"📄 **{doc['filename']}**")
            col2.caption(f"Pages: {doc['pages']}")
            col3.caption(f"Chunks: {doc['chunks']}")
            col4.caption(f"Conf: {doc['confidence']:.0f}%")
            if col5.button("🗑️", key=f"del_{doc['id']}", help="Delete document"):
                try:
                    response = asyncio.run(api_request("DELETE", f"/api/documents/{doc['id']}"))
                    if response.status_code == 200:
                        st.session_state.uploaded_docs.pop(i)
                        st.rerun()
                    else:
                        st.error(f"Delete failed: {response.text[:50]}")
                except Exception as e:
                    st.error(f"Error: {str(e)[:50]}")


def show_generate_page():
    st.title("📝 Generate Draft")

    if not st.session_state.uploaded_docs:
        st.warning("⚠️ Upload documents first!")
        return

    doc_options = {doc["filename"]: doc["id"] for doc in st.session_state.uploaded_docs}
    selected = st.multiselect(
        "Select documents",
        options=list(doc_options.keys()),
        default=list(doc_options.keys()),
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_area("Query", placeholder="What do you want to generate?", height=100)
    with col2:
        draft_type = st.selectbox("Draft Type", [
            "case_fact_summary",
            "title_review_summary",
            "notice_summary",
            "document_checklist",
            "internal_memo",
        ])

    if st.button("Generate", type="primary"):
        if not query:
            st.error("Enter a query!")
        elif not selected:
            st.error("Select at least one document!")
        else:
            with st.spinner("Generating..."):
                try:
                    payload = {
                        "query": query,
                        "draft_type": draft_type,
                        "document_ids": [doc_options[name] for name in selected],
                    }
                    response = asyncio.run(api_request("POST", "/api/drafts/generate", json=payload))

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.current_draft = result
                        st.session_state.edit_content = ""

                        st.success("✅ Draft generated")
                        col1, col2 = st.columns(2)
                        col1.metric("Type", result.get("draft_type", "N/A"))
                        col2.metric("Confidence", f"{result['confidence']:.0%}")

                        st.text_area(
                            "Draft Content",
                            value=result["content"],
                            height=300,
                            key="draft_output",
                        )

                        if result.get("citations"):
                            with st.expander(f"📚 Citations ({len(result['citations'])})"):
                                for i, c in enumerate(result["citations"], 1):
                                    st.markdown(f"**[{i}]** {c['source_doc']}, Page {c['page']}")

                        if result.get("applied_rules"):
                            with st.expander(f"🧠 Applied Rules ({len(result['applied_rules'])})"):
                                for r in result["applied_rules"]:
                                    st.markdown(f"- **WHEN:** {r['when']}")
                                    st.markdown(f"  **THEN:** {r['then']}")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Failed: {str(e)}")


def show_edit_page():
    st.title("✏️ Edit Draft")

    if st.session_state.current_draft is None:
        st.warning("⚠️ Generate a draft first!")
        return

    draft = st.session_state.current_draft
    original = draft["content"]

    st.markdown(f"**Type:** {draft.get('draft_type', 'N/A')} | **Confidence:** {draft.get('confidence', 0):.0%}")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.text_area("Original", value=original, height=350, disabled=True, label_visibility="collapsed")

    with col2:
        st.markdown("**Your Edit**")
        if not st.session_state.edit_content:
            st.session_state.edit_content = original
        edited = st.text_area(
            "Edit",
            value=st.session_state.edit_content,
            height=350,
            key="edit_area",
            label_visibility="collapsed",
        )
        st.session_state.edit_content = edited

    if edited != original:
        st.divider()
        st.info("✏️ Changes detected — submit your edit so the system can learn from it.")

        edit_reason = st.text_input("Reason (optional)", placeholder="Why this edit?")

        if st.button("Submit Edit", type="primary"):
            with st.spinner("Submitting..."):
                try:
                    payload = {
                        "draft_id": draft["draft_id"],
                        "original_text": original,
                        "edited_text": edited,
                        "edit_reason": edit_reason or None,
                        "document_context": draft.get("draft_type", ""),
                    }
                    response = asyncio.run(api_request("POST", "/api/edits/submit", json=payload))

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Submitted — Pattern: {result.get('pattern_detected', 'N/A')}")
                        st.session_state.edit_history.append({
                            "edit_id": result["edit_id"],
                            "pattern": result.get("pattern_detected"),
                            "timestamp": datetime.now().isoformat(),
                        })
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Failed: {str(e)}")


def show_learning_page():
    st.title("📊 Learning Stats")

    try:
        response = asyncio.run(api_request("GET", "/api/edits/history?limit=50"))

        if response.status_code == 200:
            history = response.json()
            edits = history.get("edits", [])

            col1, col2 = st.columns(2)
            col1.metric("Total Edits", history.get("total", 0))
            patterns = [e.get("pattern_type") for e in edits if e.get("pattern_type")]
            col2.metric("Pattern Types", len(set(patterns)) if patterns else 0)

            if patterns:
                st.divider()
                st.markdown("#### Pattern Distribution")
                pattern_counts = {}
                for p in patterns:
                    pattern_counts[p] = pattern_counts.get(p, 0) + 1
                st.bar_chart(pattern_counts)

            if edits:
                st.divider()
                st.markdown("#### Recent Edits")
                for edit in edits[:5]:
                    with st.expander(f"{edit['edit_id'][:8]}... — {edit.get('pattern_type', 'unknown')}"):
                        st.caption(f"**Original:** {edit['original_text'][:150]}...")
                        st.caption(f"**Edited:** {edit['edited_text'][:150]}...")

            if st.session_state.edit_history:
                st.divider()
                st.markdown("#### Your Edits This Session")
                for e in st.session_state.edit_history:
                    st.markdown(f"- `{e['edit_id'][:8]}...` — {e['pattern']}")
        else:
            st.warning("Could not fetch history. Backend running?")
    except Exception as e:
        st.error(f"Failed: {str(e)}")


if __name__ == "__main__":
    main()
