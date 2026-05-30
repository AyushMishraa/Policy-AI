"""
LangGraph node functions — powered by Google Gemini (free tier).

Models used:
  GEMINI_MODEL_FAST  → classify_query, validate_response  (short, structured)
  GEMINI_MODEL_GEN   → generate_response                  (long, streaming)

Each node receives the current PolicyAgentState and returns a partial
dict that LangGraph merges back into state.
"""

import json
import logging
import re
from typing import Any, Dict, List

import google.generativeai as genai

from config import config
from agent.state import PolicyAgentState

logger = logging.getLogger("agent-nodes")

# ── Initialise Gemini client once at import time ───────────────────────────
genai.configure(api_key=config.GEMINI_API_KEY)

_fast_model = genai.GenerativeModel(config.GEMINI_MODEL_FAST)
_gen_model  = genai.GenerativeModel(config.GEMINI_MODEL_GEN)


# ── Shared safety settings (disable overly cautious blocks for policy text) ─
_SAFETY = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
]


def _call_fast(prompt: str) -> str:
    """Non-streaming Gemini call for short structured outputs."""
    resp = _fast_model.generate_content(
        prompt,
        safety_settings=_SAFETY,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=512,
        ),
    )
    return resp.text.strip()


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` markdown fences from model output."""
    return re.sub(r"^```(?:json)?\s*|```$", "", text, flags=re.MULTILINE).strip()


# ═══════════════════════════════════════════════════════════════════════════
# Node 1 — classify_query
# ═══════════════════════════════════════════════════════════════════════════

async def classify_query(state: PolicyAgentState) -> Dict[str, Any]:
    """Classify intent and policy domain; rewrite query for better retrieval."""
    query = state.get("query", "")
    logger.info(f"[{state['request_id']}] Classifying: {query[:80]!r}")

    prompt = f"""You are a query classifier for a company policy assistant.
Classify the user query below and respond ONLY with valid JSON — no markdown, no explanation.

JSON schema:
{{
  "intent": <one of "lookup"|"comparison"|"summary"|"procedural"|"general">,
  "domain": <one of "hr"|"it-security"|"finance"|"legal"|"operations"|"general">,
  "refined_query": <rewritten query optimised for semantic search, max 30 words>
}}

User query: {query}"""

    try:
        raw  = _call_fast(prompt)
        data = json.loads(_strip_fences(raw))
    except Exception as exc:
        logger.warning(f"Classification failed ({exc}), using defaults")
        data = {"intent": "lookup", "domain": "general", "refined_query": query}

    logger.info(
        f"[{state['request_id']}] intent={data.get('intent')} "
        f"domain={data.get('domain')} refined={data.get('refined_query','')[:60]!r}"
    )
    return {
        "intent":             data.get("intent",        "lookup"),
        "domain":             data.get("domain",        "general"),
        "refined_query":      data.get("refined_query", query),
        "retrieval_attempts": 0,
        "regenerate_count":   0,
        "generated_text":     "",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 2 — retrieve_context
# ═══════════════════════════════════════════════════════════════════════════

async def retrieve_context(
    state: PolicyAgentState,
    vector_store,
) -> Dict[str, Any]:
    """Semantic search against Qdrant. Broadens query on retry."""
    attempts = state.get("retrieval_attempts", 0)
    query    = state.get("refined_query") or state.get("query", "")

    if attempts > 0:
        original = state.get("query", "")
        if original != query:
            query = f"{original} {query}"

    logger.info(
        f"[{state['request_id']}] Retrieving (attempt {attempts + 1}): {query[:80]!r}"
    )

    chunks = vector_store.search(query, top_k=config.TOP_K_RESULTS)
    logger.info(f"[{state['request_id']}] Retrieved {len(chunks)} chunks")

    return {
        "context_chunks":     chunks,
        "retrieval_attempts": attempts + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 3 — check_relevance
# ═══════════════════════════════════════════════════════════════════════════

RELEVANCE_THRESHOLD = 0.30


async def check_relevance(state: PolicyAgentState) -> Dict[str, Any]:
    """Flag retrieval as relevant/not based on cosine score threshold."""
    chunks    = state.get("context_chunks", [])
    top_score = max((c.get("relevance", 0) for c in chunks), default=0.0)

    if not chunks:
        return {"is_relevant": False, "relevance_reason": "No chunks returned"}

    if top_score < RELEVANCE_THRESHOLD:
        reason = f"Best score {top_score:.2f} below threshold {RELEVANCE_THRESHOLD}"
        logger.info(f"[{state['request_id']}] Not relevant — {reason}")
        return {"is_relevant": False, "relevance_reason": reason}

    logger.info(
        f"[{state['request_id']}] Relevant — top={top_score:.2f}, chunks={len(chunks)}"
    )
    return {"is_relevant": True, "relevance_reason": f"Top score {top_score:.2f}"}


# ═══════════════════════════════════════════════════════════════════════════
# Node 4 — generate_response   (streaming)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a knowledgeable and precise company policy assistant.
Answer questions ONLY based on the policy documents provided in <context>.

Rules:
- Be factual and concise.
- Quote relevant policy rules briefly when useful.
- End with a "Sources:" section listing documents used.
- If the context lacks enough information, say so clearly — do NOT invent policy details.
- Format in plain text suitable for a terminal (no markdown headers, use plain dashes for lists).
"""


def _build_context_block(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "<context>\nNo relevant policy documents found.\n</context>"
    parts = ["<context>"]
    for i, c in enumerate(chunks, 1):
        page_label = f", page {c['page']}" if c.get("page") else ""
        parts.append(f"[{i}] Source: {c['source']}{page_label}\n{c['text']}")
    parts.append("</context>")
    return "\n\n".join(parts)


def _deduplicate_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for c in chunks:
        key = (c.get("source", ""), c.get("page", ""))
        if key not in seen:
            seen.add(key)
            out.append({"source": c["source"], "page": c.get("page", "")})
    return out


async def generate_response(state: PolicyAgentState) -> Dict[str, Any]:
    """Stream a Gemini response grounded in the retrieved policy chunks."""
    query    = state.get("query", "")
    chunks   = state.get("context_chunks", [])
    is_valid = state.get("is_valid", True)
    regen    = state.get("regenerate_count", 0)

    context_block = _build_context_block(chunks)

    extra = ""
    if not is_valid and regen > 0:
        note  = state.get("validation_note", "")
        extra = (
            f"\n\nIMPORTANT — previous answer was rejected: {note}. "
            "Please generate a more accurate, grounded response."
        )

    conv_ctx   = state.get("conversation_context", "")
    conv_block = (
        f"\n<conversation_history>\n{conv_ctx}\n</conversation_history>"
        if conv_ctx else ""
    )

    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{context_block}{conv_block}\n\n"
        f"Question: {query}{extra}"
    )

    logger.info(
        f"[{state['request_id']}] Generating (regen={regen}, chunks={len(chunks)})"
    )

    full_text = ""
    try:
        response = _gen_model.generate_content(
            full_prompt,
            safety_settings=_SAFETY,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1024,
            ),
            stream=True,
        )
        for chunk in response:
            if chunk.text:
                full_text += chunk.text
    except Exception as exc:
        logger.error(f"[{state['request_id']}] Generation error: {exc}")
        return {
            "generated_text": f"Sorry, an error occurred while generating the response: {exc}",
            "sources":        [],
            "error":          str(exc),
        }

    logger.info(f"[{state['request_id']}] Generated {len(full_text)} chars")
    return {
        "generated_text": full_text,
        "sources":        _deduplicate_sources(chunks),
        "is_valid":       True,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 5 — validate_response
# ═══════════════════════════════════════════════════════════════════════════

async def validate_response(state: PolicyAgentState) -> Dict[str, Any]:
    """Lightweight hallucination / relevance check via a fast Gemini call."""
    answer      = state.get("generated_text", "")
    query       = state.get("query", "")
    chunks      = state.get("context_chunks", [])
    regen_count = state.get("regenerate_count", 0)

    # Skip validation after first regen (accept best-effort)
    if regen_count >= 1:
        logger.info(f"[{state['request_id']}] Validation skipped (max retries reached)")
        return {
            "is_valid":       True,
            "validation_note": "Accepted after max retries",
            "final_answer":   answer,
        }

    context_preview = _build_context_block(chunks[:2])

    prompt = f"""You are a quality checker for a policy assistant.
Given the query, the context, and the generated answer, decide whether the answer is:
- grounded in the provided context (not hallucinated)
- relevant to the query
- complete enough to be useful

Respond ONLY with valid JSON — no markdown:
{{"valid": true|false, "reason": "<one sentence>"}}

Query: {query}

{context_preview}

Answer:
{answer[:800]}"""

    try:
        raw  = _call_fast(prompt)
        data = json.loads(_strip_fences(raw))
        valid  = bool(data.get("valid", True))
        reason = str(data.get("reason", ""))
    except Exception as exc:
        logger.warning(f"[{state['request_id']}] Validation parse error: {exc} — accepting")
        valid, reason = True, "Validation parse error — defaulting to accept"

    logger.info(f"[{state['request_id']}] Valid={valid}  reason={reason[:80]!r}")

    if not valid:
        return {
            "is_valid":        False,
            "validation_note": reason,
            "regenerate_count": regen_count + 1,
            "generated_text":  "",
        }

    return {
        "is_valid":       True,
        "validation_note": reason,
        "final_answer":   answer,
    }
