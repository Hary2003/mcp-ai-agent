# backend/app/agent/agent.py

import os
import asyncio
from datetime import datetime

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from .mcp_servers import ACTIVE_SERVERS
from .conflict import resolve_conflicts
from .synthesis import synthesize
from app.memory.rag_memory import save_to_memory, get_context


# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------

load_dotenv(
    dotenv_path=os.path.join(
        os.path.dirname(__file__), "..", "..", "..", ".env"
    )
)

MODEL_NAME    = "llama-3.1-8b-instant"
GITHUB_REPO   = os.getenv("GITHUB_REPO", "")   # injected into prompts so the model
                                                # can call list_commits without guessing
if not GITHUB_REPO:
    print("[WARNING] GITHUB_REPO is not set — GitHub tools may fail or require manual owner/repo args.")

# Token budget
MAX_TOKENS        = 512
MAX_ITERATIONS    = 4
MAX_TOOLS         = 2
TOOL_RESULT_CHARS = 600
RAG_CONTEXT_CHARS = 300

# ------------------------------------------------------------
# Intent map — trigger SUBSTRINGS in question → preferred
# substrings to match in tool names.
#
# Rule: list tools that directly answer the question first,
# then general search tools as backup.
# ------------------------------------------------------------
_INTENT_MAP = [
    # Commits / history
    (
        ["commit", "last commit", "recent commit", "pushed", "git log",
         "latest change", "last push"],
        ["list_commit", "commit", "get_commit", "search_commit"],
    ),
    # Repositories
    (
        ["repositor", "repo", "my github", "list repo", "all repo"],
        ["list_repo", "search_repo", "get_repo"],
    ),
    # Pull requests / reviews
    (
        ["pull request", " pr ", "open pr", "merged pr", "review"],
        ["list_pull", "search_pull", "get_pull", "pull_request"],
    ),
    # GitHub issues / bugs
    (
        ["issue", "bug", "ticket", "open issue", "closed issue"],
        ["list_issue", "search_issue", "get_issue", "create_issue"],
    ),
    # Linear — deadlines / sprints
    (
        ["deadline", "due date", "due tomorrow", "due today", "overdue",
         "sprint", "milestone", "linear", "assign", "backlog",
         "blocked", "epic", "roadmap", "priority"],
        ["linear", "get_issue", "search_issue", "list_issue"],
    ),
    # Notion — pages / databases
    (
        ["notion", "page", "database", "doc", "wiki", "note", "knowledge"],
        ["notion", "search", "database", "query", "retrieve"],
    ),
    # Slack
    (
        ["slack", "channel", "thread", " dm ", "standup", "mention"],
        ["slack", "message", "channel", "search", "history"],
    ),
]

# Search tools accept a free-text "query" arg — safe to call directly.
# Structured tools (list_commits, list_issues) need named args — must go
# through the agent so the LLM can construct proper parameters.
_SEARCH_TOOL_PREFIXES = ("search_", "query_", "find_", "retrieve_")

_FALLBACK_TOOL_KEYWORDS = ["search", "query", "retrieve", "get", "list", "find"]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _make_model() -> ChatGroq:
    return ChatGroq(
        model=MODEL_NAME,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        max_tokens=MAX_TOKENS,
    )


def _truncate(text: str, max_chars: int = TOOL_RESULT_CHARS) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "…[truncated]"


def _select_tools(question: str, all_tools: list, max_tools: int = MAX_TOOLS) -> list:
    """
    Substring-based intent routing. Picks the tools most likely to answer
    this specific question, limiting schema count to keep token usage low.
    """
    q = question.lower()

    preferred_keywords: list[str] = []
    best_score = 0
    for trigger_phrases, tool_keywords in _INTENT_MAP:
        score = sum(1 for phrase in trigger_phrases if phrase in q)
        if score > best_score:
            best_score = score
            preferred_keywords = tool_keywords

    def _score_tool(tool) -> int:
        name = tool.name.lower()
        desc = (getattr(tool, "description", "") or "").lower()
        keywords = preferred_keywords or _FALLBACK_TOOL_KEYWORDS
        return sum(1 for kw in keywords if kw in name or kw in desc)

    scored = sorted(all_tools, key=_score_tool, reverse=True)
    top = [t for t in scored if _score_tool(t) > 0][:max_tools]

    if not top:
        print("[tool-select] No tools matched intent — returning empty list.")
        return []

    print(f"[tool-select] {len(all_tools)} available → matched {len(top)}: {[t.name for t in top]}")
    for t in top:
        print(f"  ↳ {t.name}: {(getattr(t, 'description', '') or '')[:80]}")
    return top


def _validate_env():
    """Warn if critical env vars are missing."""
    if not GITHUB_REPO:
        print("[WARNING] GITHUB_REPO is not set — GitHub tools may fail or require manual owner/repo args.")


def _is_search_tool(tool) -> bool:
    """
    Returns True only for free-text search tools that accept a simple
    'query' parameter. Structured tools (list_commits, list_issues) return
    False — they must go through the agent for proper arg construction.
    """
    return any(tool.name.lower().startswith(p) for p in _SEARCH_TOOL_PREFIXES)


def _system_prompt(tools: list) -> str:
    """
    Names the selected tools explicitly, injects today's date and the
    configured GitHub repo so the model can build correct tool args.
    Bans hallucinated tool calls by name.
    """
    today     = datetime.now().strftime("%A, %Y-%m-%d")
    tool_names = ", ".join(t.name for t in tools) if tools else "none"

    # Only mention GITHUB_REPO if a GitHub tool is selected
    github_hint = ""
    if GITHUB_REPO and any("github" in t.name.lower() or
                           any(k in t.name.lower() for k in
                               ["commit", "pull", "issue", "repo"])
                           for t in tools):
        owner, _, repo = GITHUB_REPO.partition("/")
        github_hint = (
            f" The configured GitHub repo is owner='{owner}' repo='{repo}'. "
            "Only use owner/repo if the tool explicitly requires them."
        )

    return (
        f"Today is {today}. "
        f"You have LIVE access to ONLY these tools: [{tool_names}].{github_hint} "
        "Use a tool only if it clearly matches the question. Use only schema-defined arguments. "
        "NEVER call tools not in the list (e.g. brave_search, web_search). "
        "NEVER say you lack access to a system. "
        "If a tool returns no results, say 'No results found via <tool_name>'. "
        "Be concise."
    )


# ------------------------------------------------------------
# Query a single MCP server  (used by ask_all)
# ------------------------------------------------------------

async def _query_single_server(server_name: str, question: str) -> str:
    """
    For search tools: direct invocation (no agent, no token cost).
    For structured tools: minimal capped agent (LLM builds the right args).
    """
    if server_name not in ACTIVE_SERVERS:
        return f"{server_name} not configured."

    try:
        client = MultiServerMCPClient({server_name: ACTIVE_SERVERS[server_name]})
        all_tools = await client.get_tools()

        tools = _select_tools(question, all_tools, max_tools=1)
        if not tools:
            return f"No suitable tool found in {server_name}."

        primary_tool = tools[0]

        # Only attempt direct invocation for search-style tools
        if _is_search_tool(primary_tool):
            for arg_key in ("query", "q", "text", "input", "search_query"):
                try:
                    raw = await primary_tool.ainvoke({arg_key: question})
                    return _truncate(str(raw))
                except Exception:
                    continue

        # Structured tools or failed search tools → agent with proper prompt
        agent = create_react_agent(_make_model(), tools)
        response = await agent.ainvoke(
            {
                "messages": [
                    {"role": "system", "content": _system_prompt(tools)},
                    {"role": "user", "content": question},
                ]
            },
            config={"recursion_limit": 2},
        )
        final = response["messages"][-1].content
        if not final or not final.strip():
            return f"Tool call to {primary_tool.name} returned no content."
        return _truncate(final)

    except Exception as e:
        print(f"MCP SERVER ERROR ({server_name}): {e}")
        return f"{server_name} error: {str(e)}"


# ------------------------------------------------------------
# Smart agent mode
# ------------------------------------------------------------

async def ask(question: str, verbose: bool = False) -> str:
    if verbose:
        print(f"\nQUESTION: {question}")

    rag_context = get_context(question)
    enhanced_question = question
    if rag_context:
        short_ctx = _truncate(rag_context, max_chars=RAG_CONTEXT_CHARS)
        enhanced_question = f"{question}\n\nContext: {short_ctx}"

    try:
        client = MultiServerMCPClient(ACTIVE_SERVERS)
        all_tools = await client.get_tools()

        if not all_tools:
            return "No MCP tools were loaded. Check MCP server configuration."

        tools = _select_tools(question, all_tools)

        if not tools:
            return "No relevant tools found for this question."

        agent = create_react_agent(_make_model(), tools)

        response = await agent.ainvoke(
            {
                "messages": [
                    {"role": "system", "content": _system_prompt(tools)},
                    {"role": "user", "content": enhanced_question},
                ]
            },
            config={"recursion_limit": MAX_ITERATIONS},
        )

        messages = response["messages"]

        # Always log tool calls for debugging
        for i, msg in enumerate(messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"[agent] Step {i} → {tc['name']}({tc['args']})")
            elif getattr(msg, "type", "") == "tool":
                status = "✓" if msg.content else "✗ empty"
                print(f"[agent] Step {i} ← {status} {str(msg.content)[:120]}")

        final_answer = messages[-1].content
        if not final_answer or not final_answer.strip():
            return "Tool call returned no usable content. Try rephrasing your question."
        save_to_memory(question=question, answer=final_answer)
        return final_answer

    except Exception as e:
        print(f"AGENT ERROR: {e}")
        return f"Agent error: {str(e)}"


# ------------------------------------------------------------
# Full parallel MCP mode
# ------------------------------------------------------------

async def ask_all(question: str, verbose: bool = False) -> str:
    start = datetime.now()
    if verbose:
        print(f"\nFULL MCP SEARCH: {question}")

    rag_context = get_context(question)
    server_names = list(ACTIVE_SERVERS.keys())

    results = await asyncio.gather(
        *[_query_single_server(name, question) for name in server_names]
    )

    raw_results = dict(zip(server_names, results))

    if verbose:
        elapsed = (datetime.now() - start).total_seconds()
        print(f"All servers responded in {elapsed:.1f}s")
        for src, result in raw_results.items():
            print(f"[{src.upper()}] {str(result)[:200]}")

    resolved, conflict_notes = resolve_conflicts(raw_results)
    final_answer = synthesize(question, resolved, conflict_notes, rag_context)

    sources_used = [k for k, v in raw_results.items() if "error" not in v.lower()]
    save_to_memory(question=question, answer=final_answer, sources_used=sources_used)

    return final_answer