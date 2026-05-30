import logging
from pathlib import Path
from typing import List, Optional

from config import config
from ingestion.loaders import (
    load_database,
    load_docx,
    load_pdf,
    load_text,
    load_url,
)

logger = logging.getLogger("pipeline")

EXTENSION_MAP = {
    ".pdf":  load_pdf,
    ".docx": load_docx,
    ".doc":  load_docx,
    ".txt":  load_text,
    ".md":   load_text,
    ".rst":  load_text,
}


class IngestionPipeline:
    def __init__(self, vector_store):
        self.vs = vector_store

    # ── Files ──────────────────────────────────────────────────────────────

    def ingest_file(self, path: str) -> int:
        p = Path(path)
        loader = EXTENSION_MAP.get(p.suffix.lower())
        if not loader:
            logger.warning(
                f"Unsupported extension '{p.suffix}' for {p.name}. "
                f"Supported: {list(EXTENSION_MAP)}"
            )
            return 0
        chunks = loader(path, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        return self.vs.add_documents(chunks) if chunks else 0

    def ingest_directory(self, dir_path: str, recursive: bool = True) -> int:
        p = Path(dir_path)
        if not p.is_dir():
            logger.error(f"Directory not found: {dir_path}")
            return 0

        pattern = "**/*" if recursive else "*"
        total = 0
        for file in sorted(p.glob(pattern)):
            if file.suffix.lower() in EXTENSION_MAP:
                n = self.ingest_file(str(file))
                total += n
                logger.info(f"  {file.name:40s} +{n} chunks")

        logger.info(f"Directory '{p.name}' complete: {total} chunks total")
        return total

    # ── URLs ───────────────────────────────────────────────────────────────

    def ingest_url(self, url: str) -> int:
        chunks = load_url(url, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        return self.vs.add_documents(chunks) if chunks else 0

    def ingest_urls(self, urls: List[str]) -> int:
        total = 0
        for url in urls:
            n = self.ingest_url(url)
            total += n
        return total

    # ── Database ───────────────────────────────────────────────────────────

    def ingest_database(
        self,
        connection_url: str,
        query: str,
        text_column: str,
        source_column: Optional[str] = None,
    ) -> int:
        chunks = load_database(
            connection_url,
            query,
            text_column,
            source_column,
            config.CHUNK_SIZE,
            config.CHUNK_OVERLAP,
        )
        return self.vs.add_documents(chunks) if chunks else 0
