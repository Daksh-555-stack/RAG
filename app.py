"""IntelliDocs AI — Streamlit frontend for the RAG chatbot.

Pipeline: upload PDFs → extract text → chunk → embed → FAISS index →
retrieve top-k chunks → LLM generates a context-grounded answer with sources.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src import embeddings, pdf_loader, qa_chain, text_splitter, utils, vector_store

# Load API keys from .env (values can still be overridden in the sidebar).
load_dotenv()

# Persisted FAISS stores live here, keyed by a hash of the uploaded files.
VECTOR_STORE_DIR = Path(__file__).resolve().parent / "data" / "vector_store"

# Provider labels shown in the sidebar radio.
HF_PROVIDER = "Hugging Face (free)"
OPENAI_PROVIDER = "OpenAI"
GROQ_PROVIDER = "Groq"

# Pre-filled model choices for the free provider.
HF_MODEL_OPTIONS = [
    qa_chain.DEFAULT_HF_MODEL,
    "google/flan-t5-large",
    "HuggingFaceH4/zephyr-7b-beta",
]


# ────────────────────────────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────────────────────────────
def init_session_state() -> None:
    """Initialise all session_state keys used by the app."""
    defaults = {
        "messages": [],           # chat history: [{role, content, sources}]
        "vector_store": None,     # FAISS index for the current document set
        "qa_chain": None,         # LCEL retrieval chain (retriever + LLM)
        "processed_files": set(), # names of successfully processed files
        "chunk_count": 0,         # number of indexed chunks
        "doc_fingerprint": None,  # hash of the processed document set
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ────────────────────────────────────────────────────────────────────────
# Sidebar: uploads, LLM settings, processing, status
# ────────────────────────────────────────────────────────────────────────
def render_provider_settings(provider: str) -> tuple[str, str, str]:
    """Render provider-specific inputs; return (api_key, hf_model, llm_model)."""
    if provider == HF_PROVIDER:
        api_key = st.text_input(
            "Hugging Face token",
            type="password",
            value=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
            help="Free token: https://huggingface.co/settings/tokens",
        )
        model_choice = st.selectbox("Model", HF_MODEL_OPTIONS + ["Custom…"])
        if model_choice == "Custom…":
            hf_model = st.text_input(
                "Custom model ID", value="", placeholder="org/model-name"
            ).strip() or qa_chain.DEFAULT_HF_MODEL
        else:
            hf_model = model_choice
        return api_key, hf_model, ""

    if provider == OPENAI_PROVIDER:
        api_key = st.text_input(
            "OpenAI API key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
        )
        model = st.text_input("Model", value="gpt-4o-mini").strip() or "gpt-4o-mini"
        return api_key, "", model

    # Groq
    api_key = st.text_input(
        "Groq API key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
    )
    model = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
    return api_key, "", model


def render_sidebar() -> None:
    """Build the sidebar: uploads → LLM config → process → status."""
    with st.sidebar:
        logo_path = Path(__file__).resolve().parent / "assets" / "logo.png"
        if logo_path.exists():
            st.image(str(logo_path), width=160)

        st.title("📄 IntelliDocs AI")
        st.caption("Ask questions about your PDFs — grounded answers with sources.")

        st.divider()
        st.subheader("1️⃣ Upload documents")
        uploaded_files = st.file_uploader(
            "Upload one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        st.divider()
        st.subheader("2️⃣ Choose the LLM")
        provider = st.radio(
            "Provider",
            [HF_PROVIDER, OPENAI_PROVIDER, GROQ_PROVIDER],
            horizontal=True,
            label_visibility="collapsed",
        )
        api_key, hf_model, llm_model = render_provider_settings(provider)

        st.divider()
        if st.button("🚀 Process documents", type="primary", use_container_width=True):
            process_documents(uploaded_files, provider, api_key, hf_model, llm_model)

        _render_status()
        if st.button("🧹 Clear session / reset", use_container_width=True):
            _clear_session()


def _render_status() -> None:
    """Show what has been indexed, plus processed file names."""
    st.divider()
    st.subheader("3️⃣ Index status")
    if st.session_state.chunk_count:
        st.metric("Chunks indexed", st.session_state.chunk_count)
    if st.session_state.processed_files:
        st.caption("Processed files:")
        for name in sorted(st.session_state.processed_files):
            st.markdown(f"- {name}")
    else:
        st.caption("No documents processed yet.")


# ────────────────────────────────────────────────────────────────────────
# Document processing
# ────────────────────────────────────────────────────────────────────────
def process_documents(
    uploaded_files: list,
    provider: str,
    api_key: str,
    hf_model: str,
    llm_model: str,
) -> None:
    """Extract → chunk → embed → build/persist the index, then wire the chain.

    Steps:
      1. Build the LLM first so missing/invalid keys fail fast with a clear message.
      2. Fingerprint the file set; reuse a persisted store if one exists.
      3. Otherwise extract, chunk, embed and persist a new FAISS index.
      4. Create the LCEL retrieval chain and store it in session state.
    """
    if not uploaded_files:
        st.warning("Please upload at least one PDF file first.")
        return

    # 1. LLM first — fail fast with a readable message.
    try:
        llm = qa_chain.build_llm(
            provider=provider,
            api_key=api_key,
            hf_model=hf_model or qa_chain.DEFAULT_HF_MODEL,
            openai_model=llm_model,
            groq_model=llm_model,
        )
    except (ValueError, RuntimeError) as exc:
        st.error(f"⚠️ LLM setup failed: {exc}")
        return

    # 2. Fingerprint the document set → maybe reuse a persisted index.
    fingerprint = utils.combined_document_hash(uploaded_files)
    index_dir = utils.index_path_for_hash(VECTOR_STORE_DIR, fingerprint)

    if fingerprint == st.session_state.doc_fingerprint and st.session_state.vector_store is not None:
        st.info("These exact documents are already indexed — reusing them.")
    elif vector_store.store_exists(index_dir):
        try:
            with st.spinner("Loading cached vector store from disk…"):
                index = vector_store.load_vector_store(index_dir, embeddings.get_embeddings())
        except (RuntimeError, FileNotFoundError) as exc:
            st.error(f"⚠️ {exc}")
            return
        st.session_state.vector_store = index
    else:
        # 3. Full pipeline: extract → chunk → embed → persist.
        try:
            embedder = embeddings.get_embeddings()
        except RuntimeError as exc:
            st.error(f"⚠️ {exc}")
            return

        with st.spinner("Extracting text from PDFs…"):
            documents, warnings = pdf_loader.extract_text_from_pdfs(uploaded_files)
        for warning in warnings:
            st.warning(warning)
        if not documents:
            st.error(
                "No readable text was extracted. The PDFs may be scanned or "
                "image-only — please upload text-based PDFs."
            )
            return

        with st.spinner(f"Chunking {len(documents)} page(s)…"):
            chunks = text_splitter.split_documents(documents)
        if not chunks:
            st.error("Chunking produced no content. Try a different PDF.")
            return

        with st.spinner("Generating embeddings and building the FAISS index…"):
            try:
                index = vector_store.create_vector_store(chunks, embedder)
                vector_store.save_vector_store(index, index_dir)
            except (ValueError, RuntimeError) as exc:
                st.error(f"⚠️ {exc}")
                return
        st.session_state.vector_store = index

    # 4. Wire up the QA chain.
    try:
        st.session_state.qa_chain = qa_chain.create_qa_chain(
            st.session_state.vector_store, llm
        )
    except RuntimeError as exc:
        st.error(f"⚠️ {exc}")
        return

    st.session_state.doc_fingerprint = fingerprint
    st.session_state.processed_files = {f.name for f in uploaded_files}
    st.session_state.chunk_count = st.session_state.vector_store.index.ntotal

    st.success(
        f"✅ Indexed {st.session_state.chunk_count} chunks from "
        f"{len(uploaded_files)} file(s). You can now ask questions!"
    )


# ────────────────────────────────────────────────────────────────────────
# Chat
# ────────────────────────────────────────────────────────────────────────
def ask_question(question: str) -> tuple[str, list[dict]]:
    """Run the question through the QA chain; return (answer, sources)."""
    chain = st.session_state.qa_chain
    if chain is None:
        return "Please upload and process documents first.", []

    try:
        with st.spinner("Searching your documents and generating an answer…"):
            # LCEL chain: invoke({"input": question}) → {"answer", "context"}.
            result = chain.invoke({"input": question})
        answer = (result.get("answer") or "").strip()
        sources = [utils.format_source(doc) for doc in result.get("context", [])]
        return answer or "I couldn't find this information in the uploaded document.", sources
    except Exception as exc:
        st.error(f"⚠️ Failed to generate an answer: {exc}")
        return f"⚠️ Something went wrong while answering: {exc}", []


def render_sources(sources: list[dict]) -> None:
    """Render an expandable 'Sources' block with filename, page and snippet."""
    for src in sources:
        with st.expander(f"📎 {src['source']} — page {src['page']}"):
            st.write(src["snippet"])


def render_chat() -> None:
    """Render conversation history, the chat input, and the empty state."""
    # History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))

    # Friendly empty state when nothing has been processed yet.
    if not st.session_state.messages:
        if st.session_state.qa_chain is None:
            st.info(
                "👋 Welcome to **IntelliDocs AI**!\n\n"
                "1. Upload one or more **PDF** files in the sidebar\n"
                "2. Pick your LLM and click **🚀 Process documents**\n"
                "3. Ask anything about the documents below",
                icon="📚",
            )
        else:
            st.info(
                "✅ Your documents are indexed — ask your first question below!",
                icon="💬",
            )

    # Input
    prompt = st.chat_input(
        "Ask a question about your documents…",
        disabled=st.session_state.qa_chain is None,
    )
    if not prompt:
        return

    # Handle the new question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer, sources = ask_question(prompt)
    with st.chat_message("assistant"):
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


# ────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="IntelliDocs AI",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    render_sidebar()
    render_chat()


def _clear_session() -> None:
    """Wipe all session state and restart the app fresh."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


if __name__ == "__main__":
    main()
