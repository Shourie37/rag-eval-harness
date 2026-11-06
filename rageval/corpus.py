"""Load markdown documents and split them into fixed-size, overlapping chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def parse_markdown(doc_id: str, raw: str) -> Document:
    """Strip heading and bullet markers so the text reads as plain sentences."""
    title = doc_id
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            if title == doc_id:
                title = heading
            # A period makes the heading its own sentence.
            lines.append(heading + ".")
        elif stripped.startswith(("- ", "* ")):
            lines.append(stripped[2:])
        else:
            lines.append(stripped)
    return Document(doc_id=doc_id, title=title, text=" ".join(lines))


def load_corpus(directory: str | Path) -> list[Document]:
    paths = sorted(Path(directory).glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no markdown files in {directory}")
    return [parse_markdown(p.stem, p.read_text(encoding="utf-8")) for p in paths]


def chunk_document(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Split a document into windows of `size` words that share `overlap` words."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    words = doc.text.split()
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        window = words[start : start + size]
        chunks.append(Chunk(f"{doc.doc_id}#c{len(chunks)}", doc.doc_id, " ".join(window)))
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks


def chunk_corpus(docs: list[Document], size: int, overlap: int) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc, size, overlap)]
