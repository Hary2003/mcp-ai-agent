# backend/app/agent/mcp_servers.py
# Only responsibility: define which MCP servers exist and how to connect to them.
# agent.py imports this. Nothing else lives here.

import os
import json
from dotenv import load_dotenv
load_dotenv()

NOTION_TOKEN    = os.getenv("NOTION_TOKEN", "")
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN", "")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_TEAM_ID   = os.getenv("SLACK_TEAM_ID", "")
LINEAR_API_KEY  = os.getenv("LINEAR_API_KEY", "")
GITHUB_REPO     = os.getenv("GITHUB_REPO", "")

ALL_SERVERS = {

    "notion": {
        "command": "npx",
        "args": ["-y", "@notionhq/notion-mcp-server"],
        "env": {
            "OPENAPI_MCP_HEADERS": json.dumps({
                "Authorization": f"Bearer {NOTION_TOKEN}",
                "Notion-Version": "2022-06-28",
            })
        },
        "transport": "stdio",
    },

    "github": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN,
            **({"GITHUB_REPO": GITHUB_REPO} if GITHUB_REPO else {}),
        },
        "transport": "stdio",
    },

    "slack": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-slack"],
        "env": {
            "SLACK_BOT_TOKEN": SLACK_BOT_TOKEN,
            "SLACK_TEAM_ID":   SLACK_TEAM_ID,
        },
        "transport": "stdio",
    },

    "linear": {
        "command": "npx",
        "args": ["-y", "@linear/mcp-server"],
        "env": {
            "LINEAR_API_KEY": LINEAR_API_KEY,
        },
        "transport": "stdio",
    },

}

# Only servers whose keys are actually set in .env
ACTIVE_SERVERS = {
    name: cfg for name, cfg in ALL_SERVERS.items()
    if all(v for v in cfg["env"].values())
}

# Authority ranking for conflict resolution — higher = more trusted
SOURCE_AUTHORITY = {
    "notion":  4,
    "linear":  3,
    "github":  2,
    "slack":   1,
}


print("\n========== ACTIVE MCP SERVERS ==========")

for name in ACTIVE_SERVERS:
    print(f"- {name}")

print("========================================\n")

print("GITHUB TOKEN:", GITHUB_TOKEN[:10] if GITHUB_TOKEN else "MISSING")