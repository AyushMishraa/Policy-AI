# Policy Agent — Async CLI AI Agent

An async, multi-session terminal AI agent for querying company policies using
**LangGraph**, **Google Gemini** (free tier), **Qdrant** vector database, and
full **voice I/O** (Whisper STT + pyttsx3 TTS).

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.0 Flash (free tier, no billing required) |
| Orchestration | LangGraph — 5-node stateful agent graph |
| Vector DB | Qdrant (Docker container, gRPC + REST) |
| Embeddings | sentence-transformers — runs fully locally |
| Server | asyncio TCP — priority queue + worker pool |
| Voice STT | faster-whisper (Whisper, local, no API) |
| Voice TTS | pyttsx3 (offline, cross-platform) |
| Terminal UI | Rich |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Terminal Sessions                            │
│   Session A  ──┐                                                     │
│   Session B  ──┼──►  Async TCP Server (asyncio)                     │
│   Session C  ──┘         │                                           │
│                          ▼                                           │
│                  Priority Queue (asyncio.PriorityQueue)              │
│                  HIGH=1 / NORMAL=5 / LOW=10                          │
│                          │                                           │
│                          ▼                                           │
│                  Worker Pool  (N async workers)                      │
│                          │                                           │
│                          ▼                                           │
│              ┌─────── LangGraph Agent ────────┐                     │
│              │  classify → retrieve → check   │                     │
│              │    → generate → validate       │                     │
│              └──────────────┬─────────────────┘                     │
│                             ▼                                        │
│                    Qdrant Vector DB (Docker)                         │
│               sentence-transformers embeddings                       │
│               (PDF / DOCX / TXT / URLs / Database)                  │
│                                                                      │
│   Voice layer (client-side)                                          │
│   STT: faster-whisper (local)   TTS: pyttsx3 (offline)             │
└──────────────────────────────────────────────────────────────────────┘
```

### LangGraph Flow

```
START → classify_query → retrieve_context → check_relevance
                                ▲                  │
                                │  retry (max 2×)  │ not relevant
                                └──────────────────┘
                                         │ relevant
                                         ▼
                                generate_response ◄─── retry (max 1×)
                                         │                    │
                                         ▼               invalid answer
                                validate_response ───────────┘
                                         │ valid
                                         ▼
                                   Stream to client
```

Each node returns a partial state dict that LangGraph merges — nodes never
share mutable state directly. `generated_text` uses `Annotated[str, operator.add]`
so streaming tokens accumulate correctly across the graph.

---

## Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- A free Google Gemini API key — get one at [aistudio.google.com](https://aistudio.google.com/app/apikey)

---

## Quick Start

### 1. Clone and install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Qdrant

```bash
docker compose up qdrant -d
```

Qdrant's dashboard will be available at `http://localhost:6333/dashboard`.

### 3. Configure

```bash
cp .env.example .env
# Open .env and set your Gemini API key:
#   GEMINI_API_KEY=AIza...
```

### 4. Ingest your policy documents

```bash
# Ingest the bundled sample policies
python main.py ingest --dir ./policies

# Or point to your own documents:
python main.py ingest --file ./docs/hr.pdf ./docs/handbook.docx
python main.py ingest --url https://company.com/policies/hr
python main.py ingest --db-url postgresql://user:pass@host/db \
    --db-query "SELECT title, content FROM policies" \
    --db-text-col content --db-src-col title
```

Ingestion is idempotent — re-running it upserts by chunk ID, never duplicates.

### 5. Start the server

```bash
python main.py server
```

### 6. Connect clients (separate terminals)

```bash
# Open as many as you like — all handled concurrently
python main.py client
```

### 7. Quick local test (no server needed)

```bash
python main.py shell
```

---

## Client Commands

| Command  | Description |
|----------|-------------|
| `/voice` | Toggle voice input — mic → Whisper STT |
| `/tts`   | Toggle TTS output — pyttsx3 reads the reply aloud |
| `/stats` | Show live server queue & session statistics |
| `/help`  | List all commands |
| `/quit`  | Disconnect and exit |

### Priority prefixes

Prepend to any query to control queue position:

```
!h What is the leave policy?        ← HIGH priority  (jumps the queue)
!l Summarise all IT policies         ← LOW priority
What is the password policy?        ← NORMAL priority (default)
```

### Voice mode

```
/voice          → activates mic recording
[press Enter]   → starts recording; pause for 1.5 s to stop
/tts            → server responses are read aloud after each reply
```

---

## Configuration Reference (`.env`)

### Gemini

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required.** Free key from aistudio.google.com |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Default model alias |
| `GEMINI_MODEL_FAST` | `gemini-2.0-flash` | Used for classify + validate (short calls) |
| `GEMINI_MODEL_GEN` | `gemini-2.0-flash` | Used for response generation (streaming) |

Free-tier model options (no billing card required):

| Model | Speed | Quality | Best for |
|---|---|---|---|
| `gemini-2.0-flash` | Fastest | Great | Recommended default |
| `gemini-1.5-flash` | Fast | Great | Alternative |
| `gemini-1.5-flash-8b` | Very fast | Good | High-volume / low-latency |

### Qdrant

| Variable | Default | Description |
|---|---|---|
| `QDRANT_HOST` | `localhost` | Qdrant host (`qdrant` inside Docker Compose) |
| `QDRANT_PORT` | `6333` | REST API port |
| `QDRANT_GRPC_PORT` | `6334` | gRPC port (used for bulk ops) |
| `QDRANT_API_KEY` | _(empty)_ | Leave empty for local Docker (no auth) |
| `QDRANT_COLLECTION` | `company_policies` | Collection name |

### Embeddings

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model (local) |
| `VECTOR_SIZE` | `384` | Must match the embedding model output size |

Common models and their vector sizes:

| Model | `VECTOR_SIZE` | Notes |
|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Default — fast, good quality |
| `all-mpnet-base-v2` | 768 | Higher quality, slower |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | Tuned specifically for Q&A |

### RAG

| Variable | Default | Description |
|---|---|---|
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `CHUNK_SIZE` | `800` | Words per chunk during ingestion |
| `CHUNK_OVERLAP` | `150` | Overlap words between chunks |
| `MIN_RELEVANCE` | `0.25` | Minimum cosine score — lower chunks are filtered |

### Server

| Variable | Default | Description |
|---|---|---|
| `SERVER_HOST` | `127.0.0.1` | TCP bind address |
| `SERVER_PORT` | `8765` | TCP port |
| `MAX_WORKERS` | `5` | Concurrent LangGraph workers |
| `QUEUE_MAX_SIZE` | `100` | Max queued requests before backpressure |

### Voice

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | Whisper size: `tiny` / `base` / `small` / `medium` / `large` |
| `TTS_ENABLED` | `true` | Enable TTS at startup |
| `TTS_VOICE_RATE` | `175` | Speaking speed (words per minute) |

---

## Adding New Policy Sources

### PDF / DOCX / TXT / Markdown

Drop files into `./policies/` and run:

```bash
python main.py ingest --dir ./policies
```

### Specific files

```bash
python main.py ingest --file ./hr.pdf ./it-handbook.docx ./remote-work.md
```

### URLs (web scraping)

```bash
python main.py ingest --url https://example.com/hr-policy https://example.com/it-policy
```

### SQL database

```bash
python main.py ingest \
  --db-url postgresql://user:pass@host/mydb \
  --db-query "SELECT title, body FROM policy_docs WHERE active = 1" \
  --db-text-col body \
  --db-src-col title
```

Supports any SQLAlchemy-compatible database: PostgreSQL, MySQL, SQLite, MSSQL.

---

## Admin CLI

Manage the Qdrant index without touching the running server:

```bash
python admin.py stats                            # collection statistics + Qdrant host
python admin.py list                             # list all indexed sources + chunk counts
python admin.py search "password requirements"   # test semantic search with scores
python admin.py delete --source hr_policy.md    # remove one source document
python admin.py delete --all                     # wipe the entire collection (with confirmation)
python admin.py reindex --dir ./policies         # delete + re-ingest a directory
python admin.py export --out sources.json        # export source list to JSON
```

---

## Docker

### Full stack (Qdrant + server)

```bash
cp .env.example .env          # set GEMINI_API_KEY
docker compose up --build -d  # starts qdrant + server
```

### Start only Qdrant (run server locally)

```bash
docker compose up qdrant -d
python main.py server         # server runs locally, connects to Dockerised Qdrant
```

### Ingest policies inside Docker

```bash
docker compose run --rm ingest
```

### Admin commands inside Docker

```bash
docker compose run --rm admin stats
docker compose run --rm admin list
docker compose run --rm admin search "leave policy"
docker compose run --rm admin delete --source old_policy.md
```

### Logs & teardown

```bash
docker compose logs -f server
docker compose logs -f qdrant
docker compose down            # stop containers (data persisted in qdrant_data volume)
docker compose down -v         # stop + delete all data
```

The `qdrant_data` Docker volume persists the vector store across restarts.
Qdrant's web dashboard is available at `http://localhost:6333/dashboard`.

---

## Project Structure

```
policy-agent/
├── main.py                  # Unified CLI — server / client / ingest / shell
├── server.py                # Async TCP server (asyncio.start_server)
├── client.py                # Rich terminal client (text + voice)
├── admin.py                 # Admin CLI for Qdrant index management
├── healthcheck.py           # TCP ping → pong (used by Docker HEALTHCHECK)
├── config.py                # Central config loaded from .env
├── requirements.txt
├── .env.example
├── Dockerfile               # Two-stage build, non-root user
├── docker-compose.yml       # qdrant + server + ingest + admin services
├── Makefile                 # Developer workflow shortcuts
├── pytest.ini
│
├── agent/
│   ├── graph.py             # LangGraph StateGraph + PolicyAgentRunner
│   ├── nodes.py             # 5 node functions powered by Gemini
│   └── state.py             # PolicyAgentState TypedDict
│
├── core/
│   ├── models.py            # QueryRequest, StreamChunk, Priority
│   ├── queue_manager.py     # asyncio.PriorityQueue + worker pool
│   ├── session_manager.py   # Per-client session registry
│   └── conversation.py      # Per-session rolling conversation history
│
├── ingestion/
│   ├── loaders.py           # PDF, DOCX, text, URL, database loaders + chunker
│   └── pipeline.py          # IngestionPipeline orchestrator
│
├── vector_store/
│   └── store.py             # Qdrant wrapper (search, upsert, delete)
│
├── voice/
│   ├── stt.py               # faster-whisper speech-to-text
│   └── tts.py               # pyttsx3 text-to-speech (thread executor)
│
├── policies/                # Drop your documents here
│   ├── hr_policy.md
│   ├── it_security_policy.md
│   └── finance_policy.md
│
└── tests/
    ├── conftest.py          # Shared fixtures (mock Gemini, mock Qdrant, sample files)
    ├── test_models.py
    ├── test_conversation.py
    ├── test_session_manager.py
    ├── test_queue_manager.py
    ├── test_loaders.py
    ├── test_pipeline.py
    ├── test_vector_store.py  # Uses in-memory Qdrant — no Docker needed
    ├── test_nodes.py         # All 5 nodes with mocked Gemini
    ├── test_graph.py         # End-to-end LangGraph routing
    └── test_admin.py         # All admin commands
```

---

## Testing

```bash
# Install dev dependencies
make dev-install

# Full test suite
make test

# Unit tests only (no I/O, very fast)
make test-unit

# With HTML coverage report → htmlcov/index.html
make test-cov

# Lint + auto-format (ruff)
make lint
make format
```

Tests never need a running Qdrant or a Gemini API key — the test suite uses:
- `QdrantClient(":memory:")` — in-memory Qdrant, no Docker required
- `monkeypatch` on `_fast_model` and `_gen_model` in `agent.nodes` — no real API calls

Test coverage:

| File | What's tested |
|---|---|
| `test_models` | Priority ordering, QueryRequest FIFO within same priority, StreamChunk fields |
| `test_conversation` | Rolling window, context string truncation, store session isolation |
| `test_session_manager` | Add / remove / touch / stats |
| `test_queue_manager` | Priority ordering, concurrent sessions, dropped disconnected sessions, error propagation |
| `test_loaders` | Chunking math, chunk_id determinism, text / MD / URL loaders |
| `test_pipeline` | File, directory (recursive + flat), URL, database ingestion paths |
| `test_vector_store` | Upsert idempotency, search top-k, relevance filter, delete by source, stats |
| `test_nodes` | All 5 LangGraph nodes — valid JSON, fallback on parse error, streaming, API error handling |
| `test_graph` | Happy path, retry-on-low-relevance loop, regenerate-on-invalid loop, conversation memory |
| `test_admin` | All 6 admin commands with patched `_init_store()` |

---

## Make Targets

```bash
make help          # list all targets with descriptions
make install       # create venv + install dependencies
make dev-install   # install + pytest + ruff
make server        # start the TCP server locally
make client        # connect a terminal client
make shell         # one-shot local query shell (no server)
make ingest        # ingest ./policies/ into Qdrant
make admin         # open the admin CLI interactively
make test          # run full test suite
make test-unit     # run unit tests only (fast)
make test-cov      # run tests + HTML coverage report
make lint          # ruff linter
make format        # ruff auto-format
make docker-build  # build the Docker image
make docker-up     # build + start qdrant + server in Docker
make docker-down   # stop all containers
make docker-logs   # follow server logs
make docker-ingest # run ingestion job inside Docker
make clean         # remove venv, __pycache__, coverage artefacts
```

---

## Extending the Agent

### Add a new LangGraph node

1. Write an async function in `agent/nodes.py` — accepts `state: PolicyAgentState`, returns a partial dict
2. Register it in `agent/graph.py` with `builder.add_node("my_node", my_fn)`
3. Wire it using `add_edge` or `add_conditional_edges`

### Change the Gemini model

Edit `GEMINI_MODEL_FAST` and `GEMINI_MODEL_GEN` in `.env`. Any model available in the Google AI Studio free tier works.

### Swap the vector store

Implement the same `search()` / `add_documents()` / `delete_source()` / `stats()` interface in a new class under `vector_store/` and inject it into `PolicyAgentRunner` in `main.py`.

### Add a new document loader

1. Write a `load_myformat(path, chunk_size, overlap) → List[dict]` function in `ingestion/loaders.py`
2. Add its extension to `EXTENSION_MAP` in `ingestion/pipeline.py`

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ConnectionRefusedError` on client | Start the server first: `python main.py server` |
| `Cannot connect to Qdrant` | Start Qdrant: `docker compose up qdrant -d` |
| `GEMINI_API_KEY not set` | Copy `.env.example` → `.env` and add your key |
| "No policy documents indexed" warning | Run `python main.py ingest --dir ./policies` |
| Qdrant `Collection not found` error | Re-run ingest — the collection is created automatically |
| Voice input not working | `pip install faster-whisper sounddevice soundfile` |
| TTS silent | `pip install pyttsx3` — also check system audio output |
| Whisper slow to load | Set `WHISPER_MODEL=tiny` in `.env` for faster startup |
| Docker healthcheck failing | Wait 40 s for startup; check `docker compose logs server` |
| Gemini `RESOURCE_EXHAUSTED` error | Free tier has per-minute limits — reduce `MAX_WORKERS` or add retry delay |
| Wrong vector size error in Qdrant | Set `VECTOR_SIZE` in `.env` to match your `EMBEDDING_MODEL` output dims |
