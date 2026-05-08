# backend/app/agent/synthesis.py
# Only responsibility: take multi-source results and synthesize one answer.
# agent.py calls synthesize() after parallel fetch + conflict resolution.

import os
from langchain_groq import ChatGroq
from .mcp_servers import SOURCE_AUTHORITY

MODEL_NAME = "llama3-groq-70b-8192-tool-use-preview"


def synthesize(
    question: str,
    raw_results: dict,
    conflict_notes: list,
    rag_context: str,
) -> str:
    """
    Combine multi-source results into one clear, attributed answer.

    Args:
        question:       The original user question
        raw_results:    Dict of {source_name: result_text}
        conflict_notes: List of conflict resolution decisions (from conflict.py)
        rag_context:    Relevant past answers from RAG memory (may be empty)

    Returns:
        A single synthesized answer string ready to show the user.
    """
    # Build the sources section in authority order
    sources_text = ""
    for source in ["notion", "linear", "github", "slack"]:
        if source in raw_results:
            authority = SOURCE_AUTHORITY.get(source, 0)
            sources_text += (
                f"\n--- {source.upper()} (authority: {authority}) ---\n"
                f"{raw_results[source]}\n"
            )

    conflict_section = ""
    if conflict_notes:
        conflict_section = "\n\nConflict resolution notes:\n" + "\n".join(
            f"• {note}" for note in conflict_notes
        )

    memory_section = ""
    if rag_context:
        memory_section = f"\n\nRelevant context from memory (past answers):\n{rag_context}"

    prompt = f"""The user asked: "{question}"

I searched internal sources simultaneously via MCP servers. Here are the results:
{sources_text}
{conflict_section}
{memory_section}

Please write a clear, well-structured answer that:
1. Directly answers the user's question
2. Attributes each piece of information to its source
   e.g. "According to Notion...", "GitHub issue #42 shows..."
3. Respects the conflict resolution notes when sources disagree
4. Uses the memory context for continuity if relevant
5. Silently skips sources that returned no results
6. Ends with a one-line summary of the most important finding
"""

    model = ChatGroq(
        model=MODEL_NAME,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )
    response = model.invoke([{"role": "user", "content": prompt}])
    return response.content