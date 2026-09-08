"""Small helper utilities: file hashing, index naming, source formatting."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


def sha256_hex(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of some bytes."""
    return hashlib.sha256(data).hexdigest()


def combined_document_hash(uploaded_files: list[Any]) -> str:
    """Fingerprint a set of uploaded files (names + contents) as one hash.

    Used to decide whether a persisted vector store can be reused: the same
    set of files always produces the same hash, so re-processing identical
    uploads skips re-embedding.

    Args:
        uploaded_files: Iterable of Streamlit ``UploadedFile`` objects.

    Returns:
        A hex SHA-256 digest covering every file's name and content.
    """
    digest = hashlib.sha256()
    for uploaded in uploaded_files:
        digest.update(uploaded.name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(uploaded.getvalue())
        digest.update(b"\x00\x00")
    return digest.hexdigest()


def index_path_for_hash(store_dir: str | Path, file_hash: str) -> Path:
    """Return the directory where a given document-set hash's index lives.

    Args:
        store_dir: Root persistence directory (default ``data/vector_store``).
        file_hash: Full hex digest from ``combined_document_hash``.

    Returns:
        A ``Path`` like ``data/vector_store/index_<first 16 hex chars>``.
    """
    return Path(store_dir) / f"index_{file_hash[:16]}"


def format_source(
    doc: Document,
    max_snippet_chars: int = 500,
) -> dict[str, Any]:
    """Turn one retrieved source Document into display-friendly info.

    Args:
        doc: A retrieved chunk (metadata: ``source`` filename, ``page`` number).
        max_snippet_chars: Cap for the quoted snippet shown in the UI.

    Returns:
        A dict with ``source``, ``page``, and a trimmed ``snippet``.
    """
    snippet = doc.page_content.strip()
    if len(snippet) > max_snippet_chars:
        snippet = snippet[:max_snippet_chars].rstrip() + "…"
    return {
        "source": doc.metadata.get("source", "unknown"),
        "page": doc.metadata.get("page", "?"),
        "snippet": snippet,
    }
