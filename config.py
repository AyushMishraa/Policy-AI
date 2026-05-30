import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Gemini ─────────────────────────────────────────────────────────────
    GEMINI_API_KEY:       str = os.getenv("GEMINI_API_KEY", "")
    # Free-tier models (no billing required)
    GEMINI_MODEL:         str = os.getenv("GEMINI_MODEL",         "gemini-2.0-flash")
    GEMINI_MODEL_FAST:    str = os.getenv("GEMINI_MODEL_FAST",    "gemini-2.0-flash")   # classify + validate
    GEMINI_MODEL_GEN:     str = os.getenv("GEMINI_MODEL_GEN",     "gemini-2.0-flash")   # generation (streaming)

    # ── Server ─────────────────────────────────────────────────────────────
    SERVER_HOST:     str = os.getenv("SERVER_HOST",     "127.0.0.1")
    SERVER_PORT:     int = int(os.getenv("SERVER_PORT", "8765"))
    MAX_WORKERS:     int = int(os.getenv("MAX_WORKERS", "5"))
    QUEUE_MAX_SIZE:  int = int(os.getenv("QUEUE_MAX_SIZE", "100"))

    # ── Qdrant ─────────────────────────────────────────────────────────────
    QDRANT_HOST:      str = os.getenv("QDRANT_HOST",      "localhost")
    QDRANT_PORT:      int = int(os.getenv("QDRANT_PORT",  "6333"))
    QDRANT_GRPC_PORT: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    QDRANT_API_KEY:   str = os.getenv("QDRANT_API_KEY",  "")  # empty = no auth (local Docker)
    COLLECTION_NAME:  str = os.getenv("QDRANT_COLLECTION", "company_policies")

    # ── Embeddings ─────────────────────────────────────────────────────────
    EMBEDDING_MODEL:  str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    VECTOR_SIZE:      int = int(os.getenv("VECTOR_SIZE", "384"))   # all-MiniLM-L6-v2 = 384

    # ── RAG ────────────────────────────────────────────────────────────────
    TOP_K_RESULTS:   int   = int(os.getenv("TOP_K_RESULTS",  "5"))
    CHUNK_SIZE:      int   = int(os.getenv("CHUNK_SIZE",     "800"))
    CHUNK_OVERLAP:   int   = int(os.getenv("CHUNK_OVERLAP",  "150"))
    MIN_RELEVANCE:   float = float(os.getenv("MIN_RELEVANCE", "0.25"))

    # ── Voice ──────────────────────────────────────────────────────────────
    WHISPER_MODEL:   str  = os.getenv("WHISPER_MODEL",   "base")
    TTS_ENABLED:     bool = os.getenv("TTS_ENABLED", "true").lower() == "true"
    TTS_VOICE_RATE:  int  = int(os.getenv("TTS_VOICE_RATE", "175"))

    # ── Paths ──────────────────────────────────────────────────────────────
    POLICIES_DIR:    str = os.getenv("POLICIES_DIR", "./policies")


config = Config()
