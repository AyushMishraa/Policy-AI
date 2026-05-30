"""
Shared state that flows through every node in the LangGraph policy agent.
Using TypedDict so LangGraph can merge partial updates from each node.
"""

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
import operator


class PolicyAgentState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────
    session_id:   str
    request_id:   str
    query:        str          # the user's raw question
    query_type:   str          # "text" | "voice"

    # ── Classification (node 1) ────────────────────────────────────────────
    intent:       str          # "lookup" | "comparison" | "summary" | "procedural" | "general"
    domain:       str          # e.g. "hr", "it-security", "finance", "legal", …
    refined_query: str         # optionally rewritten query for better retrieval

    # ── Retrieval (node 2) ─────────────────────────────────────────────────
    context_chunks: List[Dict[str, Any]]  # raw chunks from ChromaDB
    retrieval_attempts: int               # how many times we've retrieved

    # ── Relevance check (node 3) ───────────────────────────────────────────
    is_relevant:  bool         # did retrieval return useful results?
    relevance_reason: str      # short reason (logged, not shown to user)

    # ── Generation (node 4) ────────────────────────────────────────────────
    # Annotated with operator.add so partial stream tokens accumulate
    generated_text: Annotated[str, operator.add]
    sources:      List[Dict[str, Any]]   # deduplicated source references

    # ── Validation (node 5) ────────────────────────────────────────────────
    is_valid:     bool
    validation_note: str
    regenerate_count: int      # how many times we've asked to regenerate

    # ── Conversation memory ────────────────────────────────────────────────
    conversation_context: str  # prior turns injected as plain text into generation

    # ── Final output ───────────────────────────────────────────────────────
    final_answer: str
    error:        Optional[str]
