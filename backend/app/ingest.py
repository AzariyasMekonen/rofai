"""Build the retrieval index from data/sources (PDF/MD/TXT) and data/facts.md.

Usage:
    uv run python -m app.ingest            # extract, chunk, embed (if configured), save
    uv run python -m app.ingest --dry-run  # extract and chunk only; prints stats and samples
"""

import argparse
import hashlib
import re
import sys
from collections.abc import Iterator
from pathlib import Path

from pypdf import PdfReader

from .config import get_settings
from .store import Chunk, VectorStore

MAX_CHARS = 1800
OVERLAP_CHARS = 300
# Leaves room for the carried overlap (+1 for the joining newline)
UNIT_MAX_CHARS = MAX_CHARS - OVERLAP_CHARS - 1
MIN_CHUNK_CHARS = 40
SUPPORTED = {".pdf", ".md", ".txt"}

_YEAR_RE = re.compile(r"\b(19[0-9]{2}|20[0-9]{2})\b")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?።፧])\s+")


def _is_meaningful(line: str) -> bool:
    """Drop OCR debris: page numbers, stray symbols, lines that are mostly punctuation."""
    if len(line) < 3:
        return False
    useful = sum(ch.isalnum() for ch in line)
    return useful / len(line) >= 0.5


def clean_text(raw: str) -> list[str]:
    """Return cleaned paragraphs from raw OCR text."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        line = " ".join(line.split())
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if _is_meaningful(line):
            current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def _split_long(paragraph: str) -> list[str]:
    if len(paragraph) <= UNIT_MAX_CHARS:
        return [paragraph]
    pieces: list[str] = []
    buf = ""
    for sentence in _SENTENCE_SPLIT_RE.split(paragraph):
        while len(sentence) > UNIT_MAX_CHARS:
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.append(sentence[:UNIT_MAX_CHARS])
            sentence = sentence[UNIT_MAX_CHARS:]
        if buf and len(buf) + len(sentence) + 1 > UNIT_MAX_CHARS:
            pieces.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        pieces.append(buf)
    return pieces


def _overlap(buf: list[str]) -> list[str]:
    """Trailing text of a finished chunk, carried into the next so boundary-spanning facts stay retrievable."""
    carry: list[str] = []
    size = 0
    for prev in reversed(buf):
        if size + len(prev) > OVERLAP_CHARS:
            break
        carry.insert(0, prev)
        size += len(prev) + 1
    if carry:
        return carry
    tail = buf[-1][-OVERLAP_CHARS:]
    space = tail.find(" ")
    return [tail[space + 1 :] if space >= 0 else tail]


def chunk_paragraphs(paragraphs: list[str]) -> list[str]:
    units = [p for para in paragraphs for p in _split_long(para)]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for unit in units:
        if buf and size + len(unit) + 1 > MAX_CHARS:
            chunks.append("\n".join(buf))
            buf = _overlap(buf)
            size = sum(len(u) + 1 for u in buf)
        buf.append(unit)
        size += len(unit) + 1
    if buf:
        chunks.append("\n".join(buf))
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def _read_pages(path: Path) -> Iterator[tuple[int | None, str]]:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages, start=1):
            yield i, page.extract_text() or ""
    else:
        yield None, path.read_text(encoding="utf-8", errors="replace")


def _chunk_id(source: str, page: int | None, text: str) -> str:
    return hashlib.sha1(f"{source}|{page}|{text}".encode()).hexdigest()[:16]


def build_chunks(sources_dir: Path, facts_file: Path) -> list[Chunk]:
    files: list[tuple[Path, bool]] = []
    if facts_file.exists():
        files.append((facts_file, True))
    if sources_dir.exists():
        files.extend(
            (p, False) for p in sorted(sources_dir.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED
        )

    chunks: list[Chunk] = []
    for path, priority in files:
        rel = path.name if priority else str(path.relative_to(sources_dir))
        try:
            pages = list(_read_pages(path))
        except Exception as exc:  # corrupt PDFs should not abort the whole run
            print(f"  ! skipped {rel}: {exc}", file=sys.stderr)
            continue
        file_chunks = 0
        for page, raw in pages:
            for text in chunk_paragraphs(clean_text(raw)):
                years = sorted({int(y) for y in _YEAR_RE.findall(text)})
                chunks.append(Chunk(_chunk_id(rel, page, text), text, rel, page, years, priority))
                file_chunks += 1
        if file_chunks == 0:
            hint = " (is the PDF OCR'd?)" if path.suffix.lower() == ".pdf" else ""
            print(f"  ! {rel}: no usable text found{hint}", file=sys.stderr)
        else:
            print(f"  {rel}: {file_chunks} chunks")
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="extract and chunk only; no embedding calls")
    args = parser.parse_args()

    s = get_settings()
    print(f"Reading {s.facts_file} and {s.sources_dir}")
    chunks = build_chunks(s.sources_dir, s.facts_file)
    if not chunks:
        print("No chunks produced. Add OCR'd PDFs to data/sources/ or write data/facts.md.", file=sys.stderr)
        return 1
    total_chars = sum(len(c.text) for c in chunks)
    print(f"{len(chunks)} chunks, {total_chars:,} characters")

    if args.dry_run:
        for c in chunks[:3]:
            print(f"\n--- {c.source} p.{c.page} years={c.years} priority={c.priority}\n{c.text[:600]}")
        return 0

    from .llm import embed

    if not s.embeddings_configured:
        VectorStore(chunks, None, "").save(s.index_dir)
        print(f"No embedding service configured: saved a keyword-search index to {s.index_dir}")
        return 0

    print(f"Embedding with {s.embed_model} ...")
    vectors = embed([c.text for c in chunks])
    VectorStore(chunks, vectors, s.embed_model).save(s.index_dir)
    print(f"Saved index to {s.index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
