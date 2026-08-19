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

async def test_connection():
    print("Testing connection to Airtable MCP...")
    try:
        toolset = get_airtable_toolset()
        await toolset.setup()
        
        tools = toolset.get_tools()
        print(f"Success! Connected to Airtable MCP. Discovered {len(tools)} tools:")
        for t in tools:
            print(f" - {t.id}")
            
        await toolset.close()
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())