# =============================================================================
# agent_final.py — The complete system
# =============================================================================
# This is the full project as described in your spec:
#
#   ✅ All 4 platform tools: Notion, GitHub, Slack, Linear
#   ✅ Parallel query execution (asyncio.gather)
#   ✅ RAG long-term memory (past Q+A injected as context)
#   ✅ Conflict resolution (recency + source authority)
#   ✅ Response synthesis with attribution
#   ✅ Auto-saves every answer to memory
#
# Usage:
#   from agent_final import ask, ask_all
#   ask("What's the status of the auth feature?")
#   ask_all("What do we know about the payment bug?")
# =============================================================================

import os
import asyncio
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

# Import our RAG memory module
from backend.app.rag_memory import save_to_memory, get_context

load_dotenv()

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
NOTION_TOKEN    = os.getenv("NOTION_TOKEN")
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
GITHUB_REPO     = os.getenv("GITHUB_REPO")
LINEAR_API_KEY  = os.getenv("LINEAR_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing from .env")
if not NOTION_TOKEN:
    raise ValueError("NOTION_TOKEN missing from .env")

MODEL_NAME = "llama3-groq-70b-8192-tool-use-preview"


# =============================================================================
# SOURCE AUTHORITY — used in conflict resolution
# =============================================================================
# When two sources disagree, we prefer the higher-authority source.
# You can change this order based on your team's norms.
# Higher index = higher authority.

SOURCE_AUTHORITY = {
    "notion":  4,   # Official docs and wikis — most authoritative for "what should be true"
    "linear":  3,   # Tickets reflect current planned/actual state
    "github":  2,   # Code and issues — source of truth for technical state
    "slack":   1,   # Discussions — least authoritative (can be outdated quickly)
}


# =============================================================================
# TOOLS (same as v3 — copied here so this file is self-contained)
# =============================================================================

@tool
def search_notion(query: str) -> str:
    """Search Notion for pages, documentation, and notes matching the query.
    Use this for: project docs, meeting notes, specs, wikis, onboarding guides."""
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }
    try:
        r = requests.post(
            "https://api.notion.com/v1/search",
            headers=headers,
            json={"query": query, "page_size": 5},
            timeout=10,
        )
    except Exception as e:
        return f"Notion error: {e}"
    if r.status_code != 200:
        return f"Notion error {r.status_code}"
    results = r.json().get("results", [])
    if not results:
        return f"No Notion pages found for: '{query}'"
    out = []
    for item in results:
        obj_type = item.get("object", "unknown")
        url = item.get("url", "")
        date = item.get("last_edited_time", "")[:10]
        title = _notion_title(item, obj_type)
        out.append(f"• [{obj_type.upper()}] {title}  ({date})\n  {url}")
    return f"Notion — {len(results)} result(s) for '{query}':\n\n" + "\n\n".join(out)


def _notion_title(item, obj_type):
    if obj_type == "page":
        for key in ["title", "Name", "Title", "name"]:
            parts = item.get("properties", {}).get(key, {}).get("title", [])
            if parts:
                return parts[0].get("plain_text", "Untitled")
    elif obj_type == "database":
        parts = item.get("title", [])
        if parts:
            return parts[0].get("plain_text", "Untitled")
    return "Untitled"


@tool
def search_github(query: str) -> str:
    """Search GitHub for issues, pull requests, and code matching the query.
    Use this for: bug reports, feature requests, PRs, code questions, errors."""
    if not GITHUB_TOKEN:
        return "GitHub unavailable — GITHUB_TOKEN not set."
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    q = f"{query} repo:{GITHUB_REPO}" if GITHUB_REPO else query
    try:
        r = requests.get(
            "https://api.github.com/search/issues",
            headers=headers,
            params={"q": q, "per_page": 5, "sort": "updated"},
            timeout=10,
        )
    except Exception as e:
        return f"GitHub error: {e}"
    if r.status_code != 200:
        return f"GitHub error {r.status_code}"
    items = r.json().get("items", [])
    if not items:
        return f"No GitHub results for: '{query}'"
    out = []
    for item in items:
        kind = "PR" if "pull_request" in item else "ISSUE"
        state = item.get("state", "").upper()
        out.append(
            f"• [{kind} #{item['number']} {state}] {item.get('title', '')}  ({item.get('updated_at','')[:10]})\n"
            f"  {item.get('html_url','')}\n"
            f"  {(item.get('body') or '')[:120].replace(chr(10),' ')}..."
        )
    return f"GitHub — {len(items)} result(s) for '{query}':\n\n" + "\n\n".join(out)


@tool
def search_slack(query: str) -> str:
    """Search Slack for messages and conversations matching the query.
    Use this for: team discussions, decisions made in chat, announcements."""
    if not SLACK_BOT_TOKEN:
        return "Slack unavailable — SLACK_BOT_TOKEN not set."
    try:
        r = requests.get(
            "https://slack.com/api/search.messages",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            params={"query": query, "count": 5, "sort": "timestamp"},
            timeout=10,
        )
    except Exception as e:
        return f"Slack error: {e}"
    data = r.json()
    if not data.get("ok"):
        return f"Slack error: {data.get('error', 'unknown')}"
    messages = data.get("messages", {}).get("matches", [])
    if not messages:
        return f"No Slack messages for: '{query}'"
    out = []
    for msg in messages:
        try:
            from datetime import datetime as dt
            date = dt.fromtimestamp(float(msg.get("ts", 0))).strftime("%Y-%m-%d %H:%M")
        except Exception:
            date = "unknown"
        out.append(
            f"• @{msg.get('username','?')} in #{msg.get('channel',{}).get('name','?')}  ({date})\n"
            f"  \"{(msg.get('text') or '')[:200].replace(chr(10),' ')}\"\n"
            f"  {msg.get('permalink','')}"
        )
    return f"Slack — {len(messages)} message(s) for '{query}':\n\n" + "\n\n".join(out)


@tool
def search_linear(query: str) -> str:
    """Search Linear for issues, projects, and team updates matching the query.
    Use this for: project status, sprint tickets, task assignments, roadmap items."""
    if not LINEAR_API_KEY:
        return "Linear unavailable — LINEAR_API_KEY not set."
    gql = """
    query SearchIssues($query: String!) {
      issueSearch(query: $query, first: 5) {
        nodes {
          title description state { name } priority
          assignee { name } team { name } url updatedAt
          labels { nodes { name } }
        }
      }
    }
    """
    priority_map = {0: "None", 1: "Urgent", 2: "High", 3: "Medium", 4: "Low"}
    try:
        r = requests.post(
            "https://api.linear.app/graphql",
            headers={"Authorization": LINEAR_API_KEY, "Content-Type": "application/json"},
            json={"query": gql, "variables": {"query": query}},
            timeout=10,
        )
    except Exception as e:
        return f"Linear error: {e}"
    if r.status_code != 200:
        return f"Linear error {r.status_code}"
    issues = r.json().get("data", {}).get("issueSearch", {}).get("nodes", [])
    if not issues:
        return f"No Linear issues for: '{query}'"
    out = []
    for issue in issues:
        state = issue.get("state", {}).get("name", "?")
        priority = priority_map.get(issue.get("priority", 0), "?")
        assignee = (issue.get("assignee") or {}).get("name", "Unassigned")
        out.append(
            f"• [{state}] {issue.get('title','')}  ({issue.get('updatedAt','')[:10]})\n"
            f"  Priority: {priority}  Assignee: {assignee}\n"
            f"  {issue.get('url','')}\n"
            f"  {(issue.get('description') or '')[:120].replace(chr(10),' ')}..."
        )
    return f"Linear — {len(issues)} issue(s) for '{query}':\n\n" + "\n\n".join(out)


# =============================================================================
# PARALLEL FETCH — query all sources at the same time
# =============================================================================

async def _fetch_all_parallel(query: str) -> dict:
    """Fire all 4 API calls simultaneously. Much faster than sequential."""
    loop = asyncio.get_event_loop()
    results = await asyncio.gather(
        loop.run_in_executor(None, lambda: search_notion.invoke({"query": query})),
        loop.run_in_executor(None, lambda: search_github.invoke({"query": query})),
        loop.run_in_executor(None, lambda: search_slack.invoke({"query": query})),
        loop.run_in_executor(None, lambda: search_linear.invoke({"query": query})),
    )
    return {
        "notion": results[0],
        "github": results[1],
        "slack":  results[2],
        "linear": results[3],
    }


# =============================================================================
# CONFLICT RESOLUTION
# =============================================================================

def resolve_conflicts(raw_results: dict) -> tuple[dict, list[str]]:
    """
    Detect and resolve conflicts between sources.

    Current strategy:
      1. Recency — prefer the most recently updated information
      2. Authority — when recency is equal, use SOURCE_AUTHORITY ranking

    Returns:
        (resolved_results, conflict_notes) where conflict_notes lists
        any conflicts that were detected and how they were resolved.
    """
    conflict_notes = []

    # Extract timestamps from each result where possible
    # This is a simple heuristic — looks for ISO date strings in the text
    import re
    date_pattern = re.compile(r"\((\d{4}-\d{2}-\d{2})\)")

    source_dates = {}
    for source, result in raw_results.items():
        dates = date_pattern.findall(result)
        if dates:
            # Take the most recent date found in this source's results
            source_dates[source] = max(dates)

    # Check for potential conflicts:
    # If two sources mention the same keyword but with different dates,
    # flag it as a potential conflict
    keywords_to_watch = ["open", "closed", "in progress", "done", "complete", "fixed", "resolved"]

    for kw in keywords_to_watch:
        conflicting_sources = []
        for source, result in raw_results.items():
            if kw.lower() in result.lower():
                conflicting_sources.append(source)

        if len(conflicting_sources) >= 2:
            # Resolve: prefer highest authority source that has the most recent date
            best_source = max(
                conflicting_sources,
                key=lambda s: (
                    source_dates.get(s, "0000-00-00"),   # recency first
                    SOURCE_AUTHORITY.get(s, 0),           # then authority
                ),
            )
            other_sources = [s for s in conflicting_sources if s != best_source]
            if other_sources:
                note = (
                    f"Status conflict on '{kw}': found in {conflicting_sources}. "
                    f"Prioritising {best_source} "
                    f"(authority: {SOURCE_AUTHORITY.get(best_source,0)}, "
                    f"date: {source_dates.get(best_source,'unknown')})."
                )
                conflict_notes.append(note)

    return raw_results, conflict_notes


# =============================================================================
# SYNTHESIS — combine all results into one attributed answer
# =============================================================================

def synthesize(question: str, raw_results: dict, conflict_notes: list, rag_context: str) -> str:
    """
    Send all source results + RAG context to the LLM for final synthesis.

    The LLM is instructed to:
    - Combine information across sources
    - Attribute each piece to its source
    - Apply conflict resolution decisions
    - Use past memory context to fill in gaps
    """
    conflict_section = ""
    if conflict_notes:
        conflict_section = "\n\nConflict resolution notes:\n" + "\n".join(
            f"• {note}" for note in conflict_notes
        )

    memory_section = ""
    if rag_context:
        memory_section = f"\n\nRelevant context from memory (past answers):\n{rag_context}"

    prompt = f"""The user asked: "{question}"

I searched four internal sources simultaneously. Here are their raw results:

--- NOTION (authority: 4 — official docs) ---
{raw_results['notion']}

--- LINEAR (authority: 3 — project tracking) ---
{raw_results['linear']}

--- GITHUB (authority: 2 — code and issues) ---
{raw_results['github']}

--- SLACK (authority: 1 — team conversations) ---
{raw_results['slack']}
{conflict_section}
{memory_section}

Please write a clear, well-structured answer that:
1. Directly answers the user's question
2. Attributes each piece of information to its source, e.g. "According to Notion...", "GitHub issue #42 shows..."
3. Respects conflict resolution decisions when sources disagree
4. Uses the memory context to add continuity if relevant
5. Skips sources that returned no results (don't say "Slack found nothing")
6. Ends with a one-line summary of the most important finding
"""

    synth_model = ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY, temperature=0)
    response = synth_model.invoke([{"role": "user", "content": prompt}])
    return response.content


# =============================================================================
# AGENT SETUP — for ask() single-tool mode
# =============================================================================

tools = [search_notion, search_github, search_slack, search_linear]
_agent = create_react_agent(
    ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY, temperature=0),
    tools,
)


# =============================================================================
# PUBLIC API — these are the two functions you call from outside
# =============================================================================

def ask(question: str, verbose: bool = True) -> str:
    """
    Smart mode: the agent reads the question and picks the right tool(s).
    Best for focused questions that clearly target one source.

    Examples:
      ask("Is there an open GitHub issue about the login timeout?")
      ask("What does the auth spec say in Notion?")
      ask("What's the priority of ticket LIN-42?")
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")

    # Inject RAG context into the question for better continuity
    rag_context = get_context(question)
    enhanced_question = question
    if rag_context:
        enhanced_question = f"{question}\n\n[Memory context for reference:\n{rag_context}]"
        if verbose:
            print("📚 Injecting relevant memory context...")

    response = _agent.invoke({
        "messages": [{"role": "user", "content": enhanced_question}]
    })

    all_messages = response["messages"]

    if verbose:
        print("\n--- Agent reasoning ---")
        sources_called = []
        for i, msg in enumerate(all_messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n[Step {i}] Called: {tc['name']}({tc['args']})")
                    sources_called.append(tc["name"].replace("search_", ""))
            elif getattr(msg, "type", "") == "tool":
                print(f"\n[Step {i}] Result: {str(msg.content)[:200]}...")
        print("\n--- Final answer ---")

    final = all_messages[-1].content
    print(final)

    # Auto-save to memory
    save_to_memory(question=question, answer=final)
    if verbose:
        print("\n💾 Answer saved to memory.")

    return final


def ask_all(question: str, verbose: bool = True) -> str:
    """
    Full-power mode: queries ALL 4 sources in parallel, resolves conflicts,
    injects RAG memory, and synthesizes a single attributed answer.

    Best for broad questions where the answer might live anywhere.

    Examples:
      ask_all("What's the current status of the payment feature?")
      ask_all("What do we know about the auth bug?")
      ask_all("What happened with the API rate limiting issue?")
    """
    start_time = datetime.now()

    if verbose:
        print(f"\n{'='*60}")
        print(f"FULL SEARCH: {question}")
        print(f"{'='*60}")
        print("⚡ Querying all sources in parallel...")

    # Step 1: Get RAG context from memory
    rag_context = get_context(question)
    if verbose and rag_context:
        print("📚 Found relevant memory context.")

    # Step 2: Parallel fetch from all sources
    raw_results = asyncio.run(_fetch_all_parallel(question))

    if verbose:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✅ All sources responded in {elapsed:.1f}s")
        for src, result in raw_results.items():
            preview = str(result)[:100].replace("\n", " ")
            print(f"  [{src.upper()}] {preview}...")

    # Step 3: Conflict resolution
    resolved_results, conflict_notes = resolve_conflicts(raw_results)
    if verbose and conflict_notes:
        print(f"\n⚠️  Conflicts detected and resolved:")
        for note in conflict_notes:
            print(f"  {note}")

    # Step 4: Synthesize the final answer
    if verbose:
        print("\n🧠 Synthesizing answer...")

    final = synthesize(question, resolved_results, conflict_notes, rag_context)

    if verbose:
        total = (datetime.now() - start_time).total_seconds()
        print(f"\n--- Synthesised answer (total time: {total:.1f}s) ---")
    print(final)

    # Step 5: Auto-save to memory
    sources_used = [k for k, v in raw_results.items() if "No " not in v and "error" not in v.lower()]
    save_to_memory(question=question, answer=final, sources_used=sources_used)
    if verbose:
        print(f"\n💾 Answer saved to memory (sources: {sources_used})")

    return final


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("Agent Final — Complete system")
    print(f"Model: {MODEL_NAME}")
    print("Features: 4 tools + RAG memory + conflict resolution + synthesis\n")

    # Run 1: Focused question — agent picks the right tool
    ask("What Linear issues are currently marked as urgent?")

    # Run 2: Broad question — all sources queried in parallel
    ask_all("What is the current status of the authentication feature?")

    # Run 3: Ask something related — should now use memory from Run 2
    ask_all("Are there any blockers on the auth work?")