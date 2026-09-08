"""Embedding model loading.

Loads ``sentence-transformers/all-MiniLM-L6-v2`` — a small, fast, free
embedding model that runs fully **locally** (no API key, works offline after
the first download). The model is cached with ``st.cache_resource`` so it is
loaded once per session instead of on every rerun.

The heavy ``sentence_transformers``/``torch`` imports are deferred until the
function is called, keeping app startup fast.
"""

from __future__ import annotations

import streamlit as st
from langchain_core.embeddings import Embeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner="Loading the embedding model (first run downloads ~80 MB)…")
def get_embeddings(device: str = "cpu") -> Embeddings:
    """Load (and cache) the Hugging Face sentence-transformers model.

    Args:
        device: Torch device to run inference on — ``"cpu"`` by default for
            maximum compatibility; switch to ``"cuda"``/``"mps"`` for GPUs.

    Returns:
        A LangChain-compatible ``Embeddings`` instance.

    Raises:
        RuntimeError: If the model cannot be downloaded or loaded.
    """
    # Deferred import keeps app startup fast and lets the rest of the app
    # run even if torch/sentence-transformers aren't installed yet.
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "The 'langchain-huggingface' package is not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc

    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": False},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load the embedding model '{EMBEDDING_MODEL_NAME}'. "
            "Check your internet connection — the model is downloaded on "
            f"first use. Details: {exc}"
        ) from exc
