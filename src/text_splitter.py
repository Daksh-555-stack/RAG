"""Text chunking.

Splits extracted pages into overlapping segments so that retrieval can match
questions against focused snippets. ``RecursiveCharacterTextSplitter`` tries
natural break points (paragraphs, lines, sentences, words) before falling back
to raw character splits, and it carries each chunk's ``source``/``page``
metadata through unchanged.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configurable chunking constants (tune these to taste).
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def get_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Create a recursive character text splitter with overlap.

    Args:
        chunk_size: Target maximum size of each chunk, in characters.
        chunk_overlap: Number of characters shared between neighbouring
            chunks, so context that straddles a boundary isn't lost.

    Returns:
        A configured ``RecursiveCharacterTextSplitter``.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split extracted page Documents into overlapping chunks.

    Metadata (``source`` filename + ``page`` number) is preserved on every
    chunk so answers can cite their exact origin.

    Args:
        documents: Documents produced by ``pdf_loader.extract_text_from_pdfs``.
        chunk_size: Chunk size in characters (default: ``CHUNK_SIZE``).
        chunk_overlap: Overlap in characters (default: ``CHUNK_OVERLAP``).

    Returns:
        A flat list of chunk Documents with original metadata intact.
    """
    if not documents:
        return []
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(documents)
