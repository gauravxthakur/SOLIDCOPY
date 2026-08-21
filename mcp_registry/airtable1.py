import json
import os
from dotenv import load_dotenv
from livekit.agents import mcp

load_dotenv()

AIRTABLE_MCP_URL = "https://mcp.airtable.com/mcp"

async def airtable_result_resolver(ctx: mcp.MCPToolResultContext) -> str:
    """Extract and format raw Airtable MCP tool output into plain text for LiveKit."""
    if not ctx.result or not ctx.result.content:
        return "Tool executed successfully with no returned content."

    results = []
    for item in ctx.result.content:
        # Extract text directly if present
        if hasattr(item, "text") and item.text is not None:
            results.append(str(item.text))
        # Dump model dictionaries if the content item is a Pydantic/dataclass object
        elif hasattr(item, "model_dump"):
            results.append(json.dumps(item.model_dump()))
        # Fallback to string representation
        else:
            results.append(str(item))

    output = "\n".join(results).strip()
    return output if output else "Tool executed successfully with empty content."


def get_airtable_toolset() -> mcp.MCPToolset:
    """Factory function that returns a configured MCPToolset for Airtable."""
    api_key = os.getenv("AIRTABLE_PAT")
    if not api_key:
        raise ValueError("AIRTABLE_PAT environment variable is not set in environment or .env.")

    return mcp.MCPToolset(
        id="airtable",
        mcp_server=mcp.MCPServerHTTP(
            url=AIRTABLE_MCP_URL,
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
            },
            tool_result_resolver=airtable_result_resolver,
        ),
    )


async def test_connection():
    """Standalone test script to verify connection and tool discovery."""
    print("Testing connection to Airtable MCP with custom resolver...")
    toolset = get_airtable_toolset()

    try:
        await toolset.setup()
        print("Success! Connected to Airtable MCP.")
        print(f"Discovered {len(toolset.tools)} tool(s):")

        for tool in toolset.tools:
            print(f" - {tool.id}")

    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        await toolset.aclose()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_connection())