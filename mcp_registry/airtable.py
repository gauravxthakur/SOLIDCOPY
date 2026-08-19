import asyncio
import os
from livekit.agents import mcp

AIRTABLE_MCP_URL = "https://mcp.airtable.com/mcp"

def get_airtable_toolset() -> mcp.MCPToolset:
    api_key = os.getenv("AIRTABLE_PAT")
    if not api_key:
        raise ValueError("AIRTABLE_PAT environment variable is not set.")

    return mcp.MCPToolset(
        id="airtable",
        mcp_server=mcp.MCPServerHTTP(
            url=AIRTABLE_MCP_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
            },
        ),
    )