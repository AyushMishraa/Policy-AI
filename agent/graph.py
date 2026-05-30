"""
LangGraph StateGraph for the Policy Agent.

Graph topology:
  START
    → classify_query
    → retrieve_context
    → check_relevance
        ─ relevant   → generate_response
        ─ not relevant (retries left) → retrieve_context    (loop back)
        ─ not relevant (max retries)  → generate_response   (answer best-effort)
    → validate_response
        ─ valid      → END
        ─ invalid (retries left)      → generate_response   (loop back)
        ─ invalid (max retries)       → END                 (best-effort)
"""

import functools
import logging
from typing import AsyncIterator, Literal

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    check_relevance,
    classify_query,
    generate_response,
    retrieve_context,
    validate_response,
)
from agent.state import PolicyAgentState
from core.models import QueryRequest, StreamChunk

logger = logging.getLogger("agent-graph")

MAX_RETRIEVAL_RETRIES = 2
MAX_REGEN_RETRIES     = 1


# ── Routing conditions ─────────────────────────────────────────────────────

def route_after_relevance(
    state: PolicyAgentState,
) -> Literal["retrieve_context", "generate_response"]:
    if state.get("is_relevant", False):
        return "generate_response"
    attempts = state.get("retrieval_attempts", 0)
    if attempts < MAX_RETRIEVAL_RETRIES:
        logger.info(
            f"[{state.get('request_id')}] Re-retrieving "
            f"(attempt {attempts}/{MAX_RETRIEVAL_RETRIES})"
        )
        return "retrieve_context"
    logger.warning(
        f"[{state.get('request_id')}] Max retrieval retries — proceeding best-effort"
    )
    return "generate_response"


def route_after_validation(
    state: PolicyAgentState,
) -> Literal["generate_response", "__end__"]:
    if state.get("is_valid", True):
        return END
    regen = state.get("regenerate_count", 0)
    if regen <= MAX_REGEN_RETRIES:
        logger.info(
            f"[{state.get('request_id')}] Regenerating "
            f"(regen {regen}/{MAX_REGEN_RETRIES})"
        )
        return "generate_response"
    logger.warning(
        f"[{state.get('request_id')}] Max regen retries — accepting answer"
    )
    return END


# ── Graph factory ──────────────────────────────────────────────────────────

def build_graph(vector_store) -> StateGraph:
    """
    Build and compile the LangGraph StateGraph.
    vector_store is injected via functools.partial so nodes stay pure.
    """

    # Bind vector_store into retrieve_context
    _retrieve = functools.partial(retrieve_context, vector_store=vector_store)

    builder = StateGraph(PolicyAgentState)

    builder.add_node("classify_query",    classify_query)
    builder.add_node("retrieve_context",  _retrieve)
    builder.add_node("check_relevance",   check_relevance)
    builder.add_node("generate_response", generate_response)
    builder.add_node("validate_response", validate_response)

    builder.add_edge(START,              "classify_query")
    builder.add_edge("classify_query",   "retrieve_context")
    builder.add_edge("retrieve_context", "check_relevance")

    builder.add_conditional_edges(
        "check_relevance",
        route_after_relevance,
        {
            "retrieve_context":  "retrieve_context",
            "generate_response": "generate_response",
        },
    )

    builder.add_edge("generate_response", "validate_response")

    builder.add_conditional_edges(
        "validate_response",
        route_after_validation,
        {
            "generate_response": "generate_response",
            END:                 END,
        },
    )

    graph = builder.compile()
    logger.info("LangGraph compiled ✓")
    return graph


# ── Runner — yields StreamChunk objects ───────────────────────────────────

class PolicyAgentRunner:
    def __init__(self, vector_store):
        self._graph = build_graph(vector_store)

        from core.conversation import ConversationStore
        self._conv_store = ConversationStore(max_turns_per_session=8)

    async def run(self, request: QueryRequest) -> AsyncIterator[StreamChunk]:
        """
        Execute the LangGraph and yield StreamChunks suitable for the
        TCP server to forward to the terminal client.
        """
        import time
        t0 = time.time()

        # Load prior conversation turns for this session
        history = self._conv_store.get_or_create(request.session_id)
        history.add_user(request.query_text, request.request_id)
        conv_ctx = history.as_context_string()

        initial_state: PolicyAgentState = {
            "session_id":           request.session_id,
            "request_id":           request.request_id,
            "query":                request.query_text,
            "query_type":           request.query_type,
            "conversation_context": conv_ctx,
            "generated_text":       "",
        }

        final_state: PolicyAgentState = {}

        try:
            # astream_events lets us tap into each node completion
            async for event in self._graph.astream_events(
                initial_state, version="v2"
            ):
                kind = event["event"]

                # Stream partial tokens as they arrive from generate_response
                if (
                    kind == "on_chain_stream"
                    and event.get("name") == "generate_response"
                ):
                    chunk_data = event.get("data", {})
                    chunk_text = (
                        chunk_data.get("chunk", {})
                        .get("generated_text", "")
                    )
                    if chunk_text:
                        yield StreamChunk(
                            request_id=request.request_id,
                            text=chunk_text,
                        )

                # Capture final state on graph end
                if kind == "on_chain_end" and event.get("name") == "LangGraph":
                    final_state = event.get("data", {}).get("output", {})

        except Exception as exc:
            logger.exception(f"[{request.request_id}] Graph error: {exc}")
            yield StreamChunk(
                request_id=request.request_id,
                is_done=True,
                error=str(exc),
            )
            return

        # Ensure we always send the complete final answer
        # (in case streaming events didn't cover everything)
        answer  = final_state.get("final_answer") or final_state.get("generated_text", "")
        sources = final_state.get("sources", [])

        # Persist the assistant reply into conversation history
        if answer:
            history.add_assistant(answer, request.request_id)

        if answer:
            yield StreamChunk(
                request_id=request.request_id,
                text="",          # full text already streamed token-by-token
                is_done=True,
                sources=sources,
                processing_time=round(time.time() - t0, 2),
            )
        else:
            yield StreamChunk(
                request_id=request.request_id,
                is_done=True,
                error="No answer generated.",
                processing_time=round(time.time() - t0, 2),
            )
