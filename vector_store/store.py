"""
Qdrant vector store for policy chunks.

Architecture:
  - Qdrant runs as a Docker container (see docker-compose.yml)
  - sentence-transformers generates embeddings locally (no API key needed)
  - Points are upserted by a deterministic UUID derived from chunk_id
    so re-ingestion is always idempotent

Point structure:
  id      : UUID (derived from chunk_id via uuid5)
  vector  : List[float]  (sentence-transformers embedding)
  payload : {text, source, source_path, page, type, chunk_id}
"""

import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional

from config import config

logger = logging.getLogger("vector-store")


def _chunk_uuid(chunk_id: str) -> str:
    """Deterministic UUID from a chunk_id string (uuid5 in DNS namespace)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class VectorStore:
    def __init__(self):
        self._client       = None
        self._encoder      = None   # SentenceTransformer instance
        self._collection   = config.COLLECTION_NAME
        self._vector_size  = config.VECTOR_SIZE

    # ── Initialise ─────────────────────────────────────────────────────────

    async def initialize(self):
        """Connect to Qdrant and load the embedding model. Call once at startup."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from sentence_transformers import SentenceTransformer

        # ── Embedding model ─────────────────────────────────────────────
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self._encoder = SentenceTransformer(config.EMBEDDING_MODEL)

        # Auto-detect actual vector size from the model
        test_vec = self._encoder.encode(["test"])
        self._vector_size = int(test_vec.shape[1])
        logger.info(f"Embedding size: {self._vector_size}")

        # ── Qdrant client ────────────────────────────────────────────────
        kwargs: Dict[str, Any] = dict(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            grpc_port=config.QDRANT_GRPC_PORT,
            prefer_grpc=False,  # Avoid "gRPC is not available" error in some environments
            https = False,  # Disable TLS (Qdrant doesn't use it by default
        )
        if config.QDRANT_API_KEY:
            kwargs["api_key"] = config.QDRANT_API_KEY

        logger.info(f"Connecting to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}")
        self._client = QdrantClient(**kwargs)

        # ── Ensure collection exists ─────────────────────────────────────
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection '{self._collection}'")
        else:
            logger.info(f"Collection '{self._collection}' already exists")

        count = self._client.count(self._collection).count
        logger.info(f"Qdrant ready — {count} chunks indexed")

    # ── Embed ───────────────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> List[List[float]]:
        vecs = self._encoder.encode(texts, show_progress_bar=False)
        return vecs.tolist()

    # ── Search ──────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Semantic search — returns top-k chunks above the relevance threshold."""
        if self._client is None:
            return []

        total = self._client.count(self._collection).count
        if total == 0:
            return []

        k = min(top_k or config.TOP_K_RESULTS, total)

        [query_vec] = self._embed([query])

        # hits = self._client.search(
        #     collection_name=self._collection,
        #     query_vector=query_vec,
        #     limit=k,
        #     with_payload=True,
        # )
        hits = self._client.query_points(
            collection_name=self._collection,
            query=query_vec,
            limit=k,
        ).points

        chunks = []
        for hit in hits:
            payload   = hit.payload or {}
            relevance = round(float(hit.score), 3)
            if relevance < config.MIN_RELEVANCE:
                continue
            chunks.append(
                {
                    "text":      payload.get("text",        ""),
                    "source":    payload.get("source",      "Unknown"),
                    "page":      payload.get("page",        ""),
                    "type":      payload.get("type",        ""),
                    "chunk_id":  payload.get("chunk_id",    ""),
                    "relevance": relevance,
                }
            )

        return chunks

    # ── Write ────────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Upsert chunks into Qdrant. Returns number of points upserted."""
        if not documents or self._client is None:
            return 0

        from qdrant_client.models import PointStruct

        texts   = [d["text"] for d in documents]
        vectors = self._embed(texts)

        points = [
            PointStruct(
                id      = _chunk_uuid(d["chunk_id"]),
                vector  = vec,
                payload = {k: str(v) for k, v in d.items()},
            )
            for d, vec in zip(documents, vectors)
        ]

        # Qdrant upsert is idempotent — same ID overwrites
        self._client.upsert(collection_name=self._collection, points=points)
        logger.info(f"Upserted {len(points)} points into '{self._collection}'")
        return len(points)

    # ── Delete ───────────────────────────────────────────────────────────────

    def delete_source(self, source: str):
        """Remove all chunks from a specific source document."""
        if self._client is None:
            return
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
        )
        logger.info(f"Deleted points for source: '{source}'")

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        count = (
            self._client.count(self._collection).count
            if self._client else 0
        )
        return {
            "total_chunks": count,
            "collection":   self._collection,
            "qdrant_host":  f"{config.QDRANT_HOST}:{config.QDRANT_PORT}",
        }
