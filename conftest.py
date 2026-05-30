"""Shared pytest fixtures — Gemini + Qdrant edition."""

import asyncio
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Fake vector store ──────────────────────────────────────────────────────

@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.search.return_value = [
        {
            "text":      "Employees are entitled to 21 days of annual leave per year.",
            "source":    "hr_policy.md",
            "page":      "1",
            "type":      "text",
            "chunk_id":  "abc-0",
            "relevance": 0.85,
        },
        {
            "text":      "Sick leave: 12 days per year, non-transferable.",
            "source":    "hr_policy.md",
            "page":      "2",
            "type":      "text",
            "chunk_id":  "abc-1",
            "relevance": 0.71,
        },
    ]
    store.add_documents.return_value = 2
    store.stats.return_value = {
        "total_chunks": 42,
        "collection":   "company_policies",
        "qdrant_host":  "localhost:6333",
    }
    store._client = MagicMock()
    store._vector_size = 384
    store._collection  = "company_policies"
    return store


# ── Fake Gemini client ─────────────────────────────────────────────────────

@pytest.fixture
def mock_gemini(monkeypatch):
    """
    Patch google.generativeai in agent/nodes.py so no real API calls
    are made during tests.
    """
    import agent.nodes as nodes_mod

    # Default classify response
    fast_resp = MagicMock()
    fast_resp.text = '{"intent":"lookup","domain":"hr","refined_query":"annual leave entitlement"}'

    # Streaming generation response
    stream_chunk = MagicMock()
    stream_chunk.text = "You are entitled to 21 days of annual leave."

    fake_fast_model = MagicMock()
    fake_fast_model.generate_content.return_value = fast_resp

    fake_gen_model  = MagicMock()
    fake_gen_model.generate_content.return_value = iter([stream_chunk])

    monkeypatch.setattr(nodes_mod, "_fast_model", fake_fast_model)
    monkeypatch.setattr(nodes_mod, "_gen_model",  fake_gen_model)

    return fake_fast_model, fake_gen_model


# ── Sample policy text ─────────────────────────────────────────────────────

@pytest.fixture
def sample_policy_text():
    return (
        "Annual Leave Policy: All employees receive 21 days paid leave per year. "
        "Leave must be approved by the line manager at least 5 working days in advance. "
        "Unused leave may be carried forward up to 7 days. "
        "Sick Leave Policy: 12 days per year with medical certificate for absences over 3 days. "
        "Remote Work Policy: Up to 3 days per week subject to manager approval. "
    ) * 20


@pytest.fixture
def sample_policy_file(tmp_path, sample_policy_text):
    f = tmp_path / "test_policy.txt"
    f.write_text(sample_policy_text)
    return str(f)
