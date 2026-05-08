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
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        ".env"
    )
)

MODEL_NAME = "llama-3.1-8b-instant"


# ------------------------------------------------------------
# Debug helper
# ------------------------------------------------------------

def print_tools(tools):

    print("\n========== LOADED MCP TOOLS ==========")

    if not tools:
        print("No tools loaded.")

    for tool in tools:
        try:
            print(f"- {tool.name}")
        except Exception:
            print(tool)

    print("======================================\n")


# ------------------------------------------------------------
# Query a single MCP server
# ------------------------------------------------------------

async def _query_single_server(
    server_name: str,
    question: str
) -> str:

    if server_name not in ACTIVE_SERVERS:
        return f"{server_name} not configured."

    try:

        print(f"\nConnecting to MCP server: {server_name}")

        client = MultiServerMCPClient(
            {
                server_name: ACTIVE_SERVERS[server_name]
            }
        )

        all_tools = await client.get_tools()

        # Keep only lightweight/search-related tools
        tools = []

        ALLOWED_TOOL_KEYWORDS = [
            "search",
            "query",
            "retrieve",
        ]

        for tool in all_tools:

            tool_name = tool.name.lower()

            if any(
                keyword in tool_name
                for keyword in ALLOWED_TOOL_KEYWORDS
            ):
                tools.append(tool)

        print("\n========== FILTERED TOOLS ==========")

        for tool in tools:
            print(tool.name)

        print("====================================\n")

        print_tools(tools)

        if not tools:
            return f"No tools available from {server_name}"

        model = ChatGroq(
            model=MODEL_NAME,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )

        agent = create_react_agent(
            model,
            tools
        )

        response = await agent.ainvoke({
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an enterprise MCP assistant.\n"
                        "ONLY use tools that are explicitly provided.\n"
                        "Never invent tool names.\n"
                        "If tools are unavailable, clearly say so."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Search {server_name} for:\n"
                        f"{question}"
                    )
                }
            ]
        })

        return response["messages"][-1].content

    except Exception as e:

        print(f"\nMCP SERVER ERROR ({server_name})")
        print(str(e))

        return f"{server_name} error: {str(e)}"


# ------------------------------------------------------------
# Smart agent mode
# ------------------------------------------------------------

async def ask(
    question: str,
    verbose: bool = True
) -> str:

    if verbose:

        print("\n" + "=" * 60)
        print(f"QUESTION: {question}")
        print("=" * 60)

    rag_context = get_context(question)

    enhanced_question = question

    if rag_context:

        enhanced_question = (
            f"{question}\n\n"
            f"Memory Context:\n{rag_context}"
        )

        if verbose:
            print("Memory context injected.")

    try:

        client = MultiServerMCPClient(
            ACTIVE_SERVERS
        )

        tools = await client.get_tools()

        print_tools(tools)

        if not tools:
            return (
                "No MCP tools were loaded. "
                "Check MCP server configuration."
            )

        if verbose:
            print(
                f"Loaded {len(tools)} tools from "
                f"{list(ACTIVE_SERVERS.keys())}"
            )

        model = ChatGroq(
            model=MODEL_NAME,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )

        agent = create_react_agent(
            model,
            tools
        )

        response = await agent.ainvoke({
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an enterprise AI assistant.\n"
                        "ONLY use the MCP tools provided.\n"
                        "Never hallucinate tool names.\n"
                        "If a capability is unavailable, explain why."
                    )
                },
                {
                    "role": "user",
                    "content": enhanced_question
                }
            ]
        })

        messages = response["messages"]

        if verbose:

            for i, msg in enumerate(messages):

                if hasattr(msg, "tool_calls") and msg.tool_calls:

                    for tc in msg.tool_calls:

                        print(
                            f"\n[Step {i}] Tool Call:"
                        )

                        print(
                            f"Name: {tc['name']}"
                        )

                        print(
                            f"Args: {tc['args']}"
                        )

                elif getattr(msg, "type", "") == "tool":

                    print(
                        f"\n[Tool Result]"
                    )

                    print(
                        str(msg.content)[:300]
                    )

        final_answer = messages[-1].content

        print("\nFINAL ANSWER:\n")
        print(final_answer)

        save_to_memory(
            question=question,
            answer=final_answer
        )

        return final_answer

    except Exception as e:

        print("\nAGENT ERROR")
        print(str(e))

        return f"Agent error: {str(e)}"


# ------------------------------------------------------------
# Full parallel MCP mode
# ------------------------------------------------------------

async def ask_all(
    question: str,
    verbose: bool = True
) -> str:

    start = datetime.now()

    if verbose:

        print("\n" + "=" * 60)
        print(f"FULL MCP SEARCH: {question}")
        print("=" * 60)

    rag_context = get_context(question)

    server_names = list(
        ACTIVE_SERVERS.keys()
    )

    results = await asyncio.gather(
        *[
            _query_single_server(
                name,
                question
            )
            for name in server_names
        ]
    )

    raw_results = dict(
        zip(server_names, results)
    )

    if verbose:

        elapsed = (
            datetime.now() - start
        ).total_seconds()

        print(
            f"\nAll servers responded "
            f"in {elapsed:.1f}s"
        )

        for src, result in raw_results.items():

            print(
                f"\n[{src.upper()}]"
            )

            print(
                str(result)[:300]
            )

    resolved, conflict_notes = resolve_conflicts(
        raw_results
    )

    final_answer = synthesize(
        question,
        resolved,
        conflict_notes,
        rag_context
    )

    sources_used = [
        k for k, v in raw_results.items()
        if "error" not in v.lower()
    ]

    save_to_memory(
        question=question,
        answer=final_answer,
        sources_used=sources_used
    )

    return final_answer