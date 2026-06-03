"""
Document loaders for the ingestion pipeline.

Each loader returns a list of chunk dicts:
  {
    "chunk_id":    str,  # stable unique ID for upsert
    "text":        str,  # the chunk text
    "source":      str,  # display name (filename, hostname, …)
    "source_path": str,  # full path or URL
    "page":        str,  # page number or row index (may be empty)
    "type":        str,  # "pdf" | "docx" | "text" | "url" | "database"
  }
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger("loaders")


# ── Chunking helper ────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split on sentence boundaries; overlap by retaining trailing words."""
    text = text.strip()
    if not text:
        return []

    # Prefer sentence splits; fall back to paragraphs, then the whole text.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if not sentences:
        sentences = [text]

    chunks: List[str] = []
    current_words: List[str] = []

    for sent in sentences:
        sent_words = sent.split()
        if not sent_words:
            continue

        # A single sentence longer than chunk_size: hard-split by words.
        if len(sent_words) > chunk_size:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
            start = 0
            while start < len(sent_words):
                end = start + chunk_size
                chunks.append(" ".join(sent_words[start:end]))
                if end >= len(sent_words):
                    break
                start = end - overlap
            continue

        if len(current_words) + len(sent_words) > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            # Seed next chunk with the last `overlap` words for continuity.
            current_words = current_words[-overlap:]

        current_words.extend(sent_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return [c for c in chunks if c.strip()]


def _chunk_id(source: str, index: int) -> str:
    h = hashlib.md5(f"{source}:{index}".encode()).hexdigest()[:10]
    return f"{h}-{index}"


# ── PDF ────────────────────────────────────────────────────────────────────

def load_pdf(path: str, chunk_size: int = 800, overlap: int = 150) -> List[Dict[str, Any]]:
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber missing — pip install pdfplumber")
        return []

    p = Path(path)
    if not p.exists():
        logger.error(f"Not found: {path}")
        return []

    source = p.name
    chunks: List[Dict[str, Any]] = []

    try:
        with pdfplumber.open(p) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                for i, chunk in enumerate(_chunk_text(text, chunk_size, overlap)):
                    chunks.append(
                        dict(
                            chunk_id=_chunk_id(f"{source}:p{page_num}", i),
                            text=chunk,
                            source=source,
                            source_path=str(p),
                            page=str(page_num),
                            type="pdf",
                        )
                    )
    except Exception as exc:
        logger.error(f"PDF load error {path}: {exc}")

    logger.info(f"PDF  '{source}': {len(chunks)} chunks")
    return chunks


# ── DOCX ───────────────────────────────────────────────────────────────────

def load_docx(path: str, chunk_size: int = 800, overlap: int = 150) -> List[Dict[str, Any]]:
    try:
        import docx
    except ImportError:
        logger.error("python-docx missing — pip install python-docx")
        return []

    p = Path(path)
    if not p.exists():
        logger.error(f"Not found: {path}")
        return []

    try:
        doc = docx.Document(str(p))
        full_text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    except Exception as exc:
        logger.error(f"DOCX read error {path}: {exc}")
        return []

    source = p.name
    chunks = [
        dict(
            chunk_id=_chunk_id(source, i),
            text=chunk,
            source=source,
            source_path=str(p),
            page="",
            type="docx",
        )
        for i, chunk in enumerate(_chunk_text(full_text, chunk_size, overlap))
    ]

    logger.info(f"DOCX '{source}': {len(chunks)} chunks")
    return chunks


# ── Plain text / Markdown ──────────────────────────────────────────────────

def load_text(path: str, chunk_size: int = 800, overlap: int = 150) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        logger.error(f"Not found: {path}")
        return []

    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.error(f"Text read error {path}: {exc}")
        return []

    source = p.name
    chunks = [
        dict(
            chunk_id=_chunk_id(source, i),
            text=chunk,
            source=source,
            source_path=str(p),
            page="",
            type="text",
        )
        for i, chunk in enumerate(_chunk_text(text, chunk_size, overlap))
    ]

    logger.info(f"Text '{source}': {len(chunks)} chunks")
    return chunks


# ── URL ────────────────────────────────────────────────────────────────────

def load_url(url: str, chunk_size: int = 800, overlap: int = 150) -> List[Dict[str, Any]]:
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("requests/beautifulsoup4 missing — pip install requests beautifulsoup4")
        return []

    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "PolicyAgent/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
    except Exception as exc:
        logger.error(f"URL load error {url}: {exc}")
        return []

    source = urlparse(url).hostname or url
    chunks = [
        dict(
            chunk_id=_chunk_id(url, i),
            text=chunk,
            source=source,
            source_path=url,
            page="",
            type="url",
        )
        for i, chunk in enumerate(_chunk_text(text, chunk_size, overlap))
    ]

    logger.info(f"URL  '{source}': {len(chunks)} chunks")
    return chunks


# ── Database ───────────────────────────────────────────────────────────────

def load_database(
    connection_url: str,
    query: str,
    text_column: str,
    source_column: Optional[str] = None,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Dict[str, Any]]:
    """
    Fetch rows from a SQL database and chunk the text column.

    Args:
        connection_url : SQLAlchemy URL
                         e.g. postgresql://user:pass@host/db
                              sqlite:///./policies.db
        query          : SELECT statement
        text_column    : column that holds the policy text
        source_column  : column to use as the source label (optional)
    """
    try:
        from sqlalchemy import create_engine, text as sql_text
    except ImportError:
        logger.error("sqlalchemy missing — pip install sqlalchemy")
        return []

    all_chunks: List[Dict[str, Any]] = []
    rows = []
    try:
        engine = create_engine(connection_url)
        with engine.connect() as conn:
            rows = conn.execute(sql_text(query)).mappings().all()
    except Exception as exc:
        logger.error(f"DB error: {exc}")
        return []

    for row_idx, row in enumerate(rows):
        text = str(row.get(text_column, "") or "")
        source = (
            str(row.get(source_column, f"db_row_{row_idx}"))
            if source_column
            else f"db_row_{row_idx}"
        )
        # Hide credentials from stored path
        safe_url = connection_url.split("@")[-1] if "@" in connection_url else connection_url

        for i, chunk in enumerate(_chunk_text(text, chunk_size, overlap)):
            all_chunks.append(
                dict(
                    chunk_id=_chunk_id(f"{source}:{row_idx}", i),
                    text=chunk,
                    source=source,
                    source_path=safe_url,
                    page=str(row_idx),
                    type="database",
                )
            )

    logger.info(f"DB: {len(all_chunks)} chunks from {len(rows)} rows")
    return all_chunks
