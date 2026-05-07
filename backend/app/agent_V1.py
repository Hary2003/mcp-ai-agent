# =============================================================================
# agent_v2.py — Three tools + parallel execution
# =============================================================================
# What's new vs agent_v1.py:
#   ✅ search_github  — searches GitHub issues, PRs, and code
#   ✅ search_slack   — searches Slack messages across channels
#   ✅ ask_all()      — queries ALL sources in parallel (asyncio.gather)
#   ✅ ask()          — still works, agent picks tools intelligently
#
# New keys needed in .env:
#   GITHUB_TOKEN=github_pat_...
#   SLACK_BOT_TOKEN=xoxb-...
#   GITHUB_REPO=owner/repo-name    ← e.g. mycompany/backend-api
# =============================================================================

import os
import asyncio
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()

GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
NOTION_TOKEN      = os.getenv("NOTION_TOKEN")
GITHUB_TOKEN      = os.getenv("GITHUB_TOKEN")
SLACK_BOT_TOKEN   = os.getenv("SLACK_BOT_TOKEN")
GITHUB_REPO       = os.getenv("GITHUB_REPO")  # e.g. "mycompany/backend-api"

# Fail fast if critical keys are missing
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing from .env")
if not NOTION_TOKEN:
    raise ValueError("NOTION_TOKEN missing from .env")

# Warn (but don't crash) for GitHub — you can test with just Notion first.
if not GITHUB_TOKEN:
    print("WARNING: GITHUB_TOKEN not set - search_github will be unavailable")

# Slack is currently disabled for this project to keep things simple and avoid
# extra configuration / errors. The tool definition stays below but is not used.


# =============================================================================
# TOOL 1: Notion (same as v1 — no changes here)
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
        response = requests.post(
            "https://api.notion.com/v1/search",
            headers=headers,
            json={"query": query, "page_size": 2},
            timeout=10,
        )
    except Exception as e:
        return f"Notion connection error: {e}"

    if response.status_code == 401:
        return "Error: Invalid Notion token."
    if response.status_code != 200:
        return f"Notion API error {response.status_code}: {response.text}"

    results = response.json().get("results", [])
    if not results:
        return f"No Notion pages found for: '{query}'"

    formatted = []
    for item in results:
        obj_type = item.get("object", "unknown")
        url = item.get("url", "")
        last_edited = item.get("last_edited_time", "")[:10]  # just the date
        title = _get_notion_title(item, obj_type)
        formatted.append(f"• [{obj_type.upper()}] {title}  ({last_edited})\n  {url}")

    return f"Notion — {len(results)} result(s) for '{query}':\n\n" + "\n\n".join(formatted)


def _get_notion_title(item, obj_type):
    if obj_type == "page":
        props = item.get("properties", {})
        for key in ["title", "Name", "Title", "name"]:
            if key in props:
                parts = props[key].get("title", [])
                if parts:
                    return parts[0].get("plain_text", "Untitled")
    elif obj_type == "database":
        parts = item.get("title", [])
        if parts:
            return parts[0].get("plain_text", "Untitled")
    return "Untitled"


# =============================================================================
# TOOL 2: GitHub — NEW in v2
# =============================================================================
# Searches issues, pull requests, and code in your repository.
# The GitHub search API supports rich query syntax:
#   - is:issue, is:pr, is:open, is:closed
#   - label:bug, assignee:username
#   - in:title, in:body

@tool
def search_github(query: str) -> str:
    """Search GitHub for issues, pull requests, and code matching the query.
    Use this for: bug reports, feature requests, PRs, code questions, technical issues.
    Examples: 'auth bug open', 'login error is:issue', 'payment refactor is:pr'"""

    if not GITHUB_TOKEN:
        return "GitHub search unavailable — GITHUB_TOKEN not set in .env"

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    results_text = []

    # --- Search issues and PRs ---
    # GitHub's search API for issues/PRs requires at least one of:
    #   is:issue or is:pull-request
    # To avoid "Validation Failed" errors, we enforce a safe default.
    base_query = query.strip()
    if "is:issue" not in base_query and "is:pull-request" not in base_query and "is:pr" not in base_query:
        base_query = f"{base_query} is:issue"

    issue_query = base_query
    if GITHUB_REPO:
        # GITHUB_REPO should be "owner/repo", but if a full URL is set we
        # best-effort extract the "owner/repo" tail.
        repo = GITHUB_REPO
        if repo.startswith("http://") or repo.startswith("https://"):
            repo = repo.rstrip("/").split("/")[-2] + "/" + repo.rstrip("/").split("/")[-1]
        issue_query = f"{base_query} repo:{repo}"

    try:
        issue_resp = requests.get(
            "https://api.github.com/search/issues",
            headers=headers,
            params={"q": issue_query, "per_page": 5, "sort": "updated"},
            timeout=10,
        )

        if issue_resp.status_code == 200:
            items = issue_resp.json().get("items", [])
            if items:
                results_text.append(f"Issues/PRs ({len(items)} found):")
                for item in items:
                    kind = "PR" if "pull_request" in item else "ISSUE"
                    state = item.get("state", "").upper()
                    title = item.get("title", "No title")
                    url = item.get("html_url", "")
                    updated = item.get("updated_at", "")[:10]
                    body_preview = (item.get("body") or "")[:120].replace("\n", " ")
                    results_text.append(
                        f"  • [{kind} #{item['number']} {state}] {title}  ({updated})\n"
                        f"    {url}\n"
                        f"    {body_preview}..."
                    )
        elif issue_resp.status_code == 401:
            return "Error: Invalid GitHub token."
        elif issue_resp.status_code == 422:
            return f"GitHub search query invalid: {issue_resp.json().get('message', '')}"

    except Exception as e:
        results_text.append(f"GitHub issues search error: {e}")

    # --- Search code (only if repo is specified) ---
    if GITHUB_REPO:
        try:
            repo = GITHUB_REPO
            if repo.startswith("http://") or repo.startswith("https://"):
                repo = repo.rstrip("/").split("/")[-2] + "/" + repo.rstrip("/").split("/")[-1]

            code_resp = requests.get(
                "https://api.github.com/search/code",
                headers=headers,
                params={"q": f"{query} repo:{repo}", "per_page": 3},
                timeout=10,
            )

            if code_resp.status_code == 200:
                code_items = code_resp.json().get("items", [])
                if code_items:
                    results_text.append(f"\nCode matches ({len(code_items)} files):")
                    for f in code_items:
                        name = f.get("name", "")
                        path = f.get("path", "")
                        url = f.get("html_url", "")
                        results_text.append(f"  • {name}  ({path})\n    {url}")

        except Exception as e:
            results_text.append(f"GitHub code search error: {e}")

    if not results_text:
        return f"No GitHub results found for: '{query}'"

    return f"GitHub — results for '{query}':\n\n" + "\n\n".join(results_text)


# =============================================================================
# TOOL 3: Slack — NEW in v2
# =============================================================================
# Searches messages across all channels the bot has access to.
# Note: The bot needs the search:read scope to use this.
# To add the bot to a channel: /invite @your-bot-name in Slack.

@tool
def search_slack(query: str) -> str:
    """Search Slack for messages and conversations matching the query.
    Use this for: team discussions, decisions made in chat, announcements,
    quick updates, or anything that was communicated over Slack."""

    if not SLACK_BOT_TOKEN:
        return "Slack search unavailable — SLACK_BOT_TOKEN not set in .env"

    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(
            "https://slack.com/api/search.messages",
            headers=headers,
            params={"query": query, "count": 5, "sort": "timestamp"},
            timeout=10,
        )
    except Exception as e:
        return f"Slack connection error: {e}"

    if not response.ok:
        return f"Slack API HTTP error: {response.status_code}"

    data = response.json()

    # Slack returns ok:false with an error code on auth failures
    if not data.get("ok"):
        error = data.get("error", "unknown")
        if error == "not_authed":
            return "Error: Invalid Slack token."
        if error == "missing_scope":
            return "Error: Bot missing 'search:read' scope. Add it in api.slack.com/apps."
        return f"Slack API error: {error}"

    messages = data.get("messages", {}).get("matches", [])

    if not messages:
        return f"No Slack messages found for: '{query}'"

    formatted = []
    for msg in messages:
        username = msg.get("username", "Unknown user")
        channel_name = msg.get("channel", {}).get("name", "unknown-channel")
        text = (msg.get("text") or "")[:200].replace("\n", " ")
        timestamp = msg.get("ts", "")
        # Convert Slack timestamp (Unix) to readable date
        try:
            from datetime import datetime
            date_str = datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            date_str = timestamp

        permalink = msg.get("permalink", "")
        formatted.append(
            f"• @{username} in #{channel_name}  ({date_str})\n"
            f"  \"{text}\"\n"
            f"  {permalink}"
        )

    return f"Slack — {len(messages)} message(s) for '{query}':\n\n" + "\n\n".join(formatted)


# =============================================================================
# AGENT SETUP
# =============================================================================

# Use a production model. The old `llama3-groq-70b-8192-tool-use-preview`
# was decommissioned and now returns HTTP 400 model_decommissioned.
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

model = ChatGroq(
    model=MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=0,
    max_retries=1,
)

# Only use Notion + GitHub for now (Slack disabled above).
tools = [search_notion, search_github]

agent = create_react_agent(model, tools)


# =============================================================================
# ask() — Agent picks the right tools automatically (same as v1)
# =============================================================================

def ask(question: str, verbose: bool = False) -> str:

    if verbose:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")

    SYSTEM_PROMPT = """
You are an AI workspace assistant.

You have access to these tools:
- search_notion → searches internal Notion pages
- search_github → searches GitHub issues and code

IMPORTANT:
If the user mentions:
- page names
- docs
- onboarding
- notion
- internal notes
- project information

ALWAYS use search_notion first.

Never invent tools.
"""

    response = agent.invoke({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
    })

    all_messages = response["messages"]

    if verbose:
        print("\n--- Agent reasoning steps ---")

        for i, msg in enumerate(all_messages):

            if hasattr(msg, "tool_calls") and msg.tool_calls:

                for tc in msg.tool_calls:
                    print(f"\n[Step {i}] Called: {tc['name']}({tc['args']})")

            elif getattr(msg, "type", "") == "tool":

                preview = str(msg.content)[:200]

                print(f"\n[Step {i}] Result preview:\n{preview}...")

        print("\n--- Final answer ---")

    final = all_messages[-1].content

    print(final)

    return final
# =============================================================================
# ask_all() — NEW in v2: always queries all sources in parallel
# =============================================================================
# Use this when you want to search EVERYTHING at once regardless of query type.
# asyncio.gather() fires all 3 API calls simultaneously instead of one by one.
# This is significantly faster — 3 parallel calls vs 3 sequential ones.

async def _fetch_all_sources(query: str) -> dict:
    """
    Runs all three tool functions concurrently using asyncio.
    Each tool call happens at the same time — not one after another.
    """
    loop = asyncio.get_event_loop()

    # Run each blocking requests.get/post in a thread pool so they don't block each other.
    # Slack is disabled for now, so we only query Notion and GitHub.
    notion_task  = loop.run_in_executor(None, lambda: search_notion.invoke({"query": query}))
    github_task  = loop.run_in_executor(None, lambda: search_github.invoke({"query": query}))

    # Fire both simultaneously and wait for both to finish
    notion_result, github_result = await asyncio.gather(
        notion_task, github_task
    )

    return {
        "notion": notion_result,
        "github": github_result,
    }


def ask_all(question: str, verbose: bool = False) -> str:
    """
    Searches ALL sources (Notion + GitHub + Slack) in parallel, then
    sends everything to the LLM for synthesis into one answer.

    When to use ask_all() vs ask():
    - ask()     → question clearly targets one source ("find the auth bug")
    - ask_all() → broad question that might be in any source
                  ("what's the latest on the payment feature?")
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"QUESTION (all sources): {question}")
        print(f"{'='*60}")
        print("Querying Notion, GitHub, and Slack in parallel...")

    # Run the async gather in a synchronous context
    raw_results = asyncio.run(_fetch_all_sources(question))

    if verbose:
        for source, result in raw_results.items():
            preview = str(result)[:150].replace("\n", " ")
            print(f"\n[{source.upper()}] {preview}...")

    # Build a synthesis prompt with all results
    synthesis_prompt = f"""The user asked: "{question}"

I searched two internal sources simultaneously. Here are the raw results:

--- NOTION ---
{raw_results['notion']}

--- GITHUB ---
{raw_results['github']}

Please synthesize these results into a single, clear answer for the user.
- Combine related information across sources
- Note which source each piece of information came from
- If sources conflict, prefer the most recent information
- If a source returned no results, don't mention it
"""

    # Send to the LLM for synthesis (no tools needed here — just reasoning)
    from langchain_groq import ChatGroq
    synth_model = ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY, temperature=0)

    response = synth_model.invoke([{"role": "user", "content": synthesis_prompt}])
    final_answer = response.content

    if verbose:
        print("\n--- Synthesized answer ---")
    print(final_answer)
    return final_answer

if __name__ == "__main__":

    print(search_notion.invoke({
        "query": "agent_test"
    }))
# =============================================================================
# Test queries
# =============================================================================
if __name__ == "__main__":
    print(f"\nAgent v2 — Workspace Assistant")
    print(f"Model: {MODEL_NAME}\n")

    while True:
        question = input("\nYou: ")

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        ask(question, verbose=True)

    