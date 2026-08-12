# pip install "mcp[cli]" anyio

import os
import sys
import anyio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# Portable: use THIS python interpreter and the server script next to this file
# (was hardcoded Windows paths to anaconda python + D:\...\langgraph-mcp.py).
SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "langgraph-mcp.py")

server = StdioServerParameters(
    command=sys.executable,
    args=[SERVER_SCRIPT],
    env=None,
)


async def main():
    print("Starting MCP Client...")

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:

            await session.initialize()
            print("Connected to MCP Server\n")

            # 1. LIST TOOLS
            tools = await session.list_tools()
            print("Tools Available:")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")
            print()

            # 2. LIST RESOURCES
            resources = await session.list_resources()
            print(" Resources Available:")
            for res in resources.resources:
                print(f"- {res.uri}")
            print()

            # 3. LIST PROMPTS
            prompts = await session.list_prompts()
            print(" Prompts Available:")
            if prompts.prompts:
                for prompt in prompts.prompts:
                    print(f"- {prompt.name}")
            else:
                print("No prompts exposed")
            print()

            # 4. CALL TOOL
            print(" Invoking demo_query_tool...")
            result = await session.call_tool("demo_query_tool", {})
            print("Result:", result.content)
            print()

            # 5. CALL LANGGRAPH TOOL
            print(" Invoking langgraph_query_tool...")
            result = await session.call_tool(
                "langgraph_query_tool",
                {"query": "What is LangGraph?"},
            )
            print("Result:\n", result.content[:500])
            print()

            # 6. READ RESOURCE
            print(" Reading resource...")
            resource = await session.read_resource("docs://langgraph/full")
            print(resource.contents[0].text[:500])


if __name__ == "__main__":
    anyio.run(main)
