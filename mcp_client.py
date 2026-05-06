import json
import sys
import asyncio
from pathlib import Path
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = str(Path(__file__).parent / "mcp_server.py")


class MCPClient:

    def __init__(self):
        self._session = None
        self._exit_stack = None
        self.groq_tools = []
        self._tool_names = set()

    async def __aenter__(self):
        await self._connect()
        return self

    async def __aexit__(self, *_):
        if self._exit_stack:
            await self._exit_stack.aclose()

    async def _connect(self):
        self._exit_stack = AsyncExitStack()

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[SERVER_SCRIPT],
            env=None
        )

        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

        tools_response = await self._session.list_tools()
        self._build_groq_tools(tools_response.tools)

        print(f"[MCP Client] Connected. Tools: {list(self._tool_names)}")

    def _build_groq_tools(self, mcp_tools):
        self.groq_tools = []
        self._tool_names = set()
        for tool in mcp_tools:
            self.groq_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                }
            })
            self._tool_names.add(tool.name)

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        if not self._session:
            return "Error: MCP session not initialised."
        if tool_name not in self._tool_names:
            return f"Error: Unknown tool '{tool_name}'"

        print(f"[MCP Client] Calling '{tool_name}' | args: {arguments}")

        try:
            result = await self._session.call_tool(tool_name, arguments)
            text_parts = [
                block.text
                for block in result.content
                if hasattr(block, "text") and block.text
            ]
            combined = "\n".join(text_parts).strip()
            return combined or f"Tool '{tool_name}' returned no content."
        except Exception as exc:
            return f"Tool '{tool_name}' failed: {exc}"