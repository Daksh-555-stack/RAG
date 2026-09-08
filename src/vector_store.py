"""FAISS vector store: build, persist, and load.

FAISS runs fully locally (no server needed) and is fast enough for document
sets of thousands of chunks. Stores are persisted to ``data/vector_store/``
keyed by a hash of the uploaded files, so re-uploading the same documents
skips re-embedding entirely.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# data/vector_store lives next to this package (project root / data / ...).
DEFAULT_STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "vector_store"


def create_vector_store(chunks: list[Document], embeddings: Embeddings):
    """Build an in-memory FAISS index from chunked documents.

    Args:
        chunks: Chunk Documents produced by ``text_splitter.split_documents``.
        embeddings: An ``Embeddings`` instance (see ``embeddings.get_embeddings``).

    Returns:
        A ``langchain_community.vectorstores.FAISS`` index ready for retrieval.

    Raises:
        ValueError: If no chunks are provided.
        RuntimeError: If the index cannot be built.
    """
    # Deferred import keeps the module lightweight and pinpoints missing deps.
    from langchain_community.vectorstores import FAISS

    if not chunks:
        raise ValueError("No chunks provided — cannot build a vector store.")

    try:
        return FAISS.from_documents(chunks, embeddings)
    except Exception as exc:
        raise RuntimeError(f"Failed to build the FAISS index: {exc}") from exc


def save_vector_store(index, path: str | Path) -> None:
    """Persist a FAISS index to disk (writes ``index.faiss`` + ``index.pkl``).

    Args:
        index: A FAISS vector store (e.g. from ``create_vector_store``).
        path: Directory to save into; created if missing.

    Raises:
        RuntimeError: If saving fails.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    try:
        index.save_local(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to save the vector store to '{path}': {exc}") from exc


def load_vector_store(path: str | Path, embeddings: Embeddings):
    """Load a persisted FAISS index from disk.

    Note: LangChain requires ``allow_dangerous_deserialization=True`` because
    the pickled sidecar file can execute code. Only load stores that this app
    itself created (all stores under ``data/vector_store/`` qualify).

    Args:
        path: Directory containing ``index.faiss`` / ``index.pkl``.
        embeddings: The same embeddings model used when the index was saved.

    Returns:
        The loaded FAISS vector store.

    Raises:
        FileNotFoundError: If no store exists at ``path``.
        RuntimeError: If loading fails.
    """
    from langchain_community.vectorstores import FAISS

    path = Path(path)
    if not (path / "index.faiss").exists():
        raise FileNotFoundError(f"No vector store found at '{path}'.")

    try:
        return FAISS.load_local(
            str(path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load the vector store from '{path}': {exc}") from exc


def store_exists(path: str | Path) -> bool:
    """Return True if a saved FAISS store exists at ``path``."""
    return (Path(path) / "index.faiss").exists()
