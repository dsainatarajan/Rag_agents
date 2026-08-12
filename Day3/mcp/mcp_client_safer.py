# pip install "mcp[cli]" anyio

import os
import sys
import anyio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# Portable server launch (was hardcoded Windows anaconda + D:\ paths).
SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "langgraph-mcp.py")

server = StdioServerParameters(
    command=sys.executable,
    args=[SERVER_SCRIPT],
    env=None,
)

#  Timeout config (seconds)
TOOL_TIMEOUT = 10


async def safe_call_tool(session, tool_name, args):
    try:
        print(f"⚡ Calling {tool_name} with timeout {TOOL_TIMEOUT}s")
        with anyio.fail_after(TOOL_TIMEOUT):
            result = await session.call_tool(tool_name, args)
            return result
    except TimeoutError:
        print(f" TIMEOUT: {tool_name} took too long and was cancelled")
        return None
    except Exception as e:
        print(f" ERROR in {tool_name}: {str(e)}")
        return None


async def safe_read_resource(session, uri):
    try:
        print(f" Reading resource with timeout {TOOL_TIMEOUT}s")
        with anyio.fail_after(TOOL_TIMEOUT):
            resource = await session.read_resource(uri)
            return resource
    except TimeoutError:
        print(f" TIMEOUT: Resource read cancelled")
        return None
    except Exception as e:
        print(f" ERROR reading resource: {str(e)}")
        return None


async def main():
    print(" Starting MCP Client...")

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:

            await session.initialize()
            print(" Connected to MCP Server\n")

            # LIST TOOLS
            tools = await session.list_tools()
            print(" Tools Available:")
            for tool in tools.tools:
                print(f"- {tool.name}")
            print()

            # SAFE TOOL CALL
            result = await safe_call_tool(
                session,
                "langgraph_query_tool",
                {"query": "What is LangGraph?"},
            )
            if result:
                print(" Tool Response:\n", result.content[:500])
            else:
                print("️ Tool failed or timed out\n")

            # SAFE RESOURCE READ
            resource = await safe_read_resource(session, "docs://langgraph/full")
            if resource:
                print(" Resource Preview:\n", resource.contents[0].text[:500])
            else:
                print(" Resource fetch failed")


if __name__ == "__main__":
    anyio.run(main)
