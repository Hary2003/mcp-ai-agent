# =============================================================================
# rag_memory.py — Long-term memory for your agent
# =============================================================================
# What this file does:
#   - Every time the agent answers a question, we store the Q+A pair
#   - Next time a similar question is asked, we retrieve past answers as context
#   - This gives the agent "memory" across conversations
#
# How RAG works in 3 lines:
#   1. Convert text → numbers (embeddings) using a local sentence-transformer model
#   2. Store those numbers in ChromaDB (a local vector database)
#   3. On each new query, find the stored entries whose numbers are most similar
#
# New packages needed:
#   pip install chromadb sentence-transformers
#
# No API keys needed — everything runs locally and for free.
# =============================================================================

import os
import uuid
import json
from datetime import datetime
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer


# =============================================================================
# SETUP — load the embedding model and connect to ChromaDB
# =============================================================================

# Where ChromaDB stores its data on disk (persists across runs)
DB_PATH = "./agent_memory_db"

# Embedding model — converts text to vectors
# "all-MiniLM-L6-v2" is fast, small (~80MB), and good enough for this use case
# It downloads automatically the first time you run this
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# How many past memories to inject as context per query
TOP_K_RESULTS = 3

# How similar a memory must be to be included (0.0 = any, 1.0 = identical)
# 0.6 means "at least 60% similar" — tune this if you get too many irrelevant memories
SIMILARITY_THRESHOLD = 0.6

print("Loading embedding model (downloads once, then cached)...")
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)

_chroma_client = chromadb.PersistentClient(path=DB_PATH)
_collection = _chroma_client.get_or_create_collection(
    name="agent_memory",
    metadata={"hnsw:space": "cosine"},  # cosine similarity for text
)

print(f"Memory loaded. Stored entries: {_collection.count()}")


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def save_to_memory(question: str, answer: str, sources_used: Optional[list] = None) -> str:
    """
    Store a question + answer pair in the vector database.

    Call this after every successful agent response so it builds up memory
    over time. The question is embedded into a vector and stored with the
    full answer as metadata.

    Args:
        question:     The user's original question
        answer:       The agent's full answer
        sources_used: Optional list of sources that contributed (e.g. ["notion", "github"])

    Returns:
        The memory ID (useful for debugging)
    """
    memory_id = str(uuid.uuid4())

    # Embed the question (this is what we search against later)
    embedding = _embed_model.encode(question).tolist()

    # Store everything as metadata alongside the vector
    metadata = {
        "question":     question,
        "answer":       answer[:2000],  # ChromaDB has a metadata size limit
        "sources":      json.dumps(sources_used or []),
        "timestamp":    datetime.now().isoformat(),
        "date":         datetime.now().strftime("%Y-%m-%d"),
    }

    _collection.add(
        ids=[memory_id],
        embeddings=[embedding],
        documents=[question],   # the text we're indexing
        metadatas=[metadata],
    )

    return memory_id


def get_context(query: str, top_k: int = TOP_K_RESULTS) -> str:
    """
    Retrieve the most relevant past Q+A pairs for a given query.

    This is called before each new agent response. If relevant past
    memories exist, they're formatted as context and injected into
    the synthesis prompt to give the agent "memory".

    Args:
        query:  The current user question
        top_k:  How many past memories to retrieve

    Returns:
        A formatted string of past Q+A context, or empty string if none found
    """
    if _collection.count() == 0:
        return ""   # No memories yet — first run

    embedding = _embed_model.encode(query).tolist()

    results = _collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, _collection.count()),  # can't ask for more than we have
        include=["metadatas", "distances"],
    )

    memories = []
    for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
        similarity = 1 - distance  # ChromaDB cosine distance → similarity

        # Only include memories that are similar enough
        if similarity < SIMILARITY_THRESHOLD:
            continue

        q = meta.get("question", "")
        a = meta.get("answer", "")[:500]   # truncate long answers
        date = meta.get("date", "")
        sources = json.loads(meta.get("sources", "[]"))
        source_str = f" [via {', '.join(sources)}]" if sources else ""

        memories.append(
            f"Past Q ({date}{source_str}): {q}\n"
            f"Past A: {a}"
        )

    if not memories:
        return ""

    context = "=== Relevant past answers from memory ===\n\n"
    context += "\n\n---\n\n".join(memories)
    context += "\n\n==========================================\n"
    return context


def clear_memory() -> int:
    """
    Delete all stored memories. Useful for testing or resetting.
    Returns the number of entries deleted.
    """
    count = _collection.count()
    if count > 0:
        all_ids = _collection.get()["ids"]
        _collection.delete(ids=all_ids)
    print(f"Cleared {count} memories.")
    return count


def memory_stats() -> dict:
    """Return a summary of what's in memory — useful for debugging."""
    count = _collection.count()
    if count == 0:
        return {"total": 0, "entries": []}

    all_data = _collection.get(include=["metadatas"])
    entries = [
        {
            "date": m.get("date", ""),
            "question": m.get("question", "")[:80],
            "sources": json.loads(m.get("sources", "[]")),
        }
        for m in all_data["metadatas"]
    ]
    return {"total": count, "entries": entries}


# =============================================================================
# TEST — run this file directly to verify RAG is working
# =============================================================================

if __name__ == "__main__":
    print("\n--- Testing RAG memory ---\n")

    # 1. Save some test memories
    print("Saving 3 test memories...")
    save_to_memory(
        question="What is the authentication flow?",
        answer="The auth flow uses JWT tokens. Users log in via /api/auth/login, get a token, and include it in the Authorization header for subsequent requests. Tokens expire after 24 hours.",
        sources_used=["notion", "github"],
    )
    save_to_memory(
        question="How do we handle payment errors?",
        answer="Payment errors are caught in the PaymentService class. Stripe webhook failures are logged to Datadog and retried 3 times. The team discussed this in #payments-backend on 2024-01-15.",
        sources_used=["slack", "github"],
    )
    save_to_memory(
        question="What is the deployment process?",
        answer="We use GitHub Actions for CI/CD. PRs to main trigger a build pipeline. Staging deploys automatically; production requires manual approval. See the deployment runbook in Notion.",
        sources_used=["notion"],
    )
    print(f"Saved. Total memories: {_collection.count()}\n")

    # 2. Retrieve context for a related query
    test_query = "How does login work in our system?"
    print(f"Retrieving context for: '{test_query}'")
    context = get_context(test_query)
    print(context if context else "No relevant memories found.")

    # 3. Print memory stats
    print("\n--- Memory stats ---")
    stats = memory_stats()
    print(f"Total stored: {stats['total']}")
    for entry in stats["entries"]:
        print(f"  • [{entry['date']}] {entry['question']}... (sources: {entry['sources']})")