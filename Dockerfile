# ══════════════════════════════════════════════════════════════════════════
# Stage 1 — builder: install Python deps into a clean venv
# ══════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /build

# System libs needed by pdfplumber, sounddevice, pyttsx3
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libffi-dev libssl-dev \
        portaudio19-dev \
        espeak-ng \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated venv
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ══════════════════════════════════════════════════════════════════════════
# Stage 2 — runtime: copy only what's needed
# ══════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Policy Agent"
LABEL org.opencontainers.image.description="Async CLI AI agent for company policies"

# Runtime system libs (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
        portaudio19-dev \
        espeak-ng \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd -m -u 1000 agent
WORKDIR /app

# Copy venv from builder
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy application code
COPY --chown=agent:agent . .

# Persistent volume mount points
RUN mkdir -p /app/data/chroma_db /app/policies && \
    chown -R agent:agent /app/data /app/policies

USER agent

# Expose TCP server port
EXPOSE 8765

# Health-check: ping the server every 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python healthcheck.py

# Default command: start the server
CMD ["python", "main.py", "server"]
