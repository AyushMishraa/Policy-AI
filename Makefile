# ══════════════════════════════════════════════════════════════════════════
# Policy Agent — Makefile
# Usage:  make <target>
# ══════════════════════════════════════════════════════════════════════════

.PHONY: help install dev-install test test-cov lint format \
        server client shell ingest admin \
        docker-build docker-up docker-down docker-logs \
        docker-ingest docker-admin clean

PYTHON    ?= python3
VENV      ?= .venv
PIP       ?= $(VENV)/bin/pip
PYTEST    ?= $(VENV)/bin/pytest
RUFF      ?= $(VENV)/bin/ruff

# ── Help ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Policy Agent — available targets"
	@echo ""
	@echo "  Setup"
	@echo "    install        Create venv and install dependencies"
	@echo "    dev-install    install + dev tools (pytest, ruff)"
	@echo ""
	@echo "  Run (local)"
	@echo "    server         Start the async TCP server"
	@echo "    client         Start a terminal client session"
	@echo "    shell          One-shot local query shell (no server)"
	@echo "    ingest         Ingest ./policies/ into the vector store"
	@echo "    admin          Open the admin CLI (interactive)"
	@echo ""
	@echo "  Tests"
	@echo "    test           Run the full test suite"
	@echo "    test-cov       Run tests + generate HTML coverage report"
	@echo "    lint           Run ruff linter"
	@echo "    format         Auto-format with ruff"
	@echo ""
	@echo "  Docker"
	@echo "    docker-build   Build the Docker image"
	@echo "    docker-up      Build + start server container"
	@echo "    docker-down    Stop and remove containers"
	@echo "    docker-logs    Follow server logs"
	@echo "    docker-ingest  Run the ingest job inside Docker"
	@echo "    docker-admin   Open admin CLI inside Docker"
	@echo ""
	@echo "    clean          Remove venv, __pycache__, coverage artefacts"
	@echo ""

# ── Setup ──────────────────────────────────────────────────────────────────
install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "  ✓  Venv ready.  Copy .env.example → .env and set ANTHROPIC_API_KEY"
	@echo ""

dev-install: install
	$(PIP) install pytest pytest-asyncio pytest-cov ruff
	@echo "  ✓  Dev tools installed."

# ── Local run ──────────────────────────────────────────────────────────────
server:
	$(VENV)/bin/python main.py server

client:
	$(VENV)/bin/python main.py client

shell:
	$(VENV)/bin/python main.py shell

ingest:
	$(VENV)/bin/python main.py ingest --dir ./policies

admin:
	$(VENV)/bin/python admin.py

# ── Tests ──────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v

test-unit:
	$(PYTEST) tests/ -v -m "not integration and not slow"

test-cov:
	$(PYTEST) tests/ --cov=. --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "  Coverage report → htmlcov/index.html"
	@echo ""

lint:
	$(RUFF) check .

format:
	$(RUFF) check --fix .
	$(RUFF) format .

# ── Docker ─────────────────────────────────────────────────────────────────
docker-build:
	docker compose build

docker-up:
	docker compose up --build -d server
	@echo ""
	@echo "  Server running on port $${SERVER_PORT:-8765}"
	@echo "  Connect: python main.py client   (from local venv)"
	@echo ""

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f server

docker-ingest:
	docker compose run --rm ingest

docker-admin:
	docker compose run --rm admin $(filter-out $@,$(MAKECMDGOALS))

# ── Clean ──────────────────────────────────────────────────────────────────
clean:
	rm -rf $(VENV) htmlcov .coverage .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	@echo "  ✓  Cleaned."
