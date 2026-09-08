"""PDF text extraction.

Reads PDFs from raw bytes (as produced by Streamlit's ``st.file_uploader``),
extracts per-page text with ``pypdf``, and returns LangChain ``Document``
objects carrying ``source``/``page`` metadata so chunks can later be cited
back to a specific page of a specific file.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from langchain_core.documents import Document
from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> list[Document]:
    """Extract the text of a single PDF into one Document per page.

    Args:
        file_bytes: Raw contents of the PDF file.
        filename: Display name of the uploaded file (used as the source).

    Returns:
        A list of Documents, one per page with text, each carrying metadata
        ``{"source": filename, "page": page_number}`` (1-indexed).

    Raises:
        ValueError: If the file cannot be parsed as a PDF, or if no
            extractable text is found (e.g. a scanned / image-only PDF).
    """
    # Parse the PDF from memory — no temp files needed.
    try:
        reader = PdfReader(BytesIO(file_bytes))
    except Exception as exc:  # pypdf raises several different error types
        raise ValueError(f"Could not read '{filename}' as a PDF: {exc}") from exc

    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as exc:  # a single broken page shouldn't kill the file
            text = ""
        if not text:
            continue  # image-only page — skip it (a warning is raised below)
        documents.append(
            Document(
                page_content=text,
                metadata={"source": filename, "page": page_number},
            )
        )

    if not documents:
        raise ValueError(
            f"No extractable text was found in '{filename}'. It may be a "
            "scanned or image-only PDF — try a text-based PDF instead."
        )
    return documents


def extract_text_from_pdfs(uploaded_files: list[Any]) -> tuple[list[Document], list[str]]:
    """Extract Documents from many uploaded files, failing gracefully.

    One bad file (wrong type, scanned, corrupt) produces a warning instead of
    aborting the whole batch.

    Args:
        uploaded_files: Iterable of Streamlit ``UploadedFile`` objects.

    Returns:
        A tuple of ``(documents, warnings)`` where ``warnings`` is a list of
        human-readable messages describing any files that were skipped.
    """
    all_documents: list[Document] = []
    warnings: list[str] = []

    for uploaded in uploaded_files:
        filename: str = uploaded.name or "unnamed.pdf"
        if not filename.lower().endswith(".pdf"):
            warnings.append(f"Skipped '{filename}': only PDF files are supported.")
            continue
        try:
            all_documents.extend(extract_text_from_pdf(uploaded.getvalue(), filename))
        except ValueError as exc:
            warnings.append(str(exc))
        except Exception as exc:  # defensive: never crash the batch
            warnings.append(f"Unexpected error while processing '{filename}': {exc}")

    return all_documents, warnings
