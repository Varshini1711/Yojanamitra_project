"""
mcp_server.py — Yojana Mitra MCP Server
========================================
Exposes two tools via MCP stdio protocol:
  • web_search(query)  → DuckDuckGo search, .gov.in results prioritised
  • fetch_url(url)     → Clean text extraction from any HTTP URL

Run standalone:
    python mcp_server.py

The MCP Client (mcp_client.py) spawns this as a subprocess automatically.
"""

import urllib3
import requests
from bs4 import BeautifulSoup
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─────────────────────────────────────────────────────────────────────────────
server = Server("yojana-mitra-tools")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="web_search",
            description=(
                "Search the web using DuckDuckGo for information about Indian government schemes. "
                "Use when a scheme is NOT found in the local ChromaDB database, or when the user "
                "explicitly asks for latest updates, installment dates, or current news about a scheme. "
                "Always prefer .gov.in results. Returns top 5 results with title, URL, and snippet."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Precise search query. Always include scheme name and 'government scheme India'. "
                            "Example: 'PM Kisan Samman Nidhi latest installment 2025 site:pmkisan.gov.in'"
                        )
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="fetch_url",
            description=(
                "Fetch and extract full readable text from an official government website URL. "
                "Use this AFTER web_search to get complete scheme details, OR when the user asks "
                "for more details and a source URL already exists in the local database record. "
                "Returns cleaned plain text up to 3500 characters."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "Full URL starting with https:// or http://. "
                            "Prefer .gov.in URLs for scheme information."
                        )
                    }
                },
                "required": ["url"]
            }
        )
    ]


# ══════════════════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    # ── web_search ────────────────────────────────────────────────────────
    if name == "web_search":
        query = arguments.get("query", "").strip()
        if not query:
            return [types.TextContent(type="text", text="Error: query cannot be empty.")]

        print(f"[MCP Server] web_search → '{query}'")

        try:
            from ddgs import DDGS
        except ImportError:
            return [types.TextContent(
                type="text",
                text="web_search tool unavailable. Install: pip install ddgs"
            )]

        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=8))

            if not raw_results:
                return [types.TextContent(type="text", text="No search results found for this query.")]

            # Prioritise official .gov.in results
            gov_results = [r for r in raw_results if ".gov.in" in r.get("href", "")]
            other_results = [r for r in raw_results if r not in gov_results]
            ordered = (gov_results + other_results)[:5]

            lines = []
            for i, r in enumerate(ordered, 1):
                lines.append(
                    f"Result {i}:\n"
                    f"  Title   : {r.get('title', 'N/A')}\n"
                    f"  URL     : {r.get('href', 'N/A')}\n"
                    f"  Snippet : {r.get('body', 'N/A')}"
                )

            return [types.TextContent(type="text", text="\n\n".join(lines))]

        except Exception as exc:
            return [types.TextContent(type="text", text=f"web_search failed: {exc}")]

    # ── fetch_url ─────────────────────────────────────────────────────────
    elif name == "fetch_url":
        url = arguments.get("url", "").strip()
        if not url:
            return [types.TextContent(type="text", text="Error: url cannot be empty.")]
        if not url.startswith(("http://", "https://")):
            return [types.TextContent(type="text", text="Error: URL must start with http:// or https://")]

        print(f"[MCP Server] fetch_url → '{url}'")

        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            }
            resp = requests.get(url, headers=headers, timeout=12, verify=False)

            content_type = resp.headers.get("content-type", "")
            if "pdf" in content_type or "octet-stream" in content_type:
                return [types.TextContent(
                    type="text",
                    text=f"This URL returns a PDF/binary file — cannot extract text. URL: {url}"
                )]

            if resp.status_code != 200:
                return [types.TextContent(
                    type="text",
                    text=f"HTTP {resp.status_code} — failed to fetch {url}"
                )]

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "meta", "noscript"]):
                tag.decompose()

            lines = [
                line.strip()
                for line in soup.get_text("\n").splitlines()
                if line.strip() and len(line.strip()) > 20
            ]
            content = "\n".join(lines)[:3500]

            if len(content) < 80:
                return [types.TextContent(
                    type="text",
                    text=f"Could not extract useful content from {url}. The page may require JavaScript."
                )]

            print(f"[MCP Server] fetch_url → extracted {len(content)} chars")
            return [types.TextContent(type="text", text=content)]

        except Exception as exc:
            return [types.TextContent(type="text", text=f"fetch_url failed: {exc}")]

    # ── unknown tool ──────────────────────────────────────────────────────
    return [types.TextContent(type="text", text=f"Unknown tool: '{name}'")]


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    print("[MCP Server] Starting Yojana Mitra MCP Server …")
    print("[MCP Server] Tools exposed: web_search, fetch_url")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())