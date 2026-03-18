"""
src/rag/document_loader.py
───────────────────────────
Loads and chunks documents from .txt and .pdf files.
Returns a list of Document objects ready for embedding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass
class Document:
    """A text chunk with provenance metadata."""
    content: str
    source: str
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return f"Document(source={self.source!r}, chunk={self.chunk_index}, preview={preview!r})"


class DocumentLoader:
    """
    Loads .txt and .pdf files into chunked Document objects.

    Parameters
    ──────────
    chunk_size    : target character length per chunk
    chunk_overlap : number of characters of overlap between adjacent chunks
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self, path: str) -> List[Document]:
        """Load a single file (txt or pdf) and return chunks."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if p.suffix.lower() == ".pdf":
            text = self._load_pdf(p)
        else:
            text = p.read_text(encoding="utf-8", errors="replace")

        return list(self._chunk(text, source=p.name))

    def load_directory(self, directory: str, extensions: Optional[List[str]] = None) -> List[Document]:
        """Load all matching files in a directory recursively."""
        extensions = extensions or [".txt", ".pdf", ".md"]
        docs: List[Document] = []
        for p in Path(directory).rglob("*"):
            if p.is_file() and p.suffix.lower() in extensions:
                try:
                    docs.extend(self.load(str(p)))
                except Exception:
                    pass
        return docs

    def load_text(self, text: str, source: str = "inline") -> List[Document]:
        """Chunk a raw string directly."""
        return list(self._chunk(text, source=source))

    # ── Internal ───────────────────────────────────────────────────────────────

    def _chunk(self, text: str, source: str) -> Iterator[Document]:
        """Sliding-window character chunker."""
        # Normalise whitespace
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            return

        start = 0
        chunk_idx = 0
        step = max(1, self.chunk_size - self.chunk_overlap)

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                yield Document(
                    content=chunk_text,
                    source=source,
                    chunk_index=chunk_idx,
                )
                chunk_idx += 1
            start += step

    @staticmethod
    def _load_pdf(path: Path) -> str:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            raise ImportError("pip install pypdf")
        reader = PdfReader(str(path))
        return "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        )
