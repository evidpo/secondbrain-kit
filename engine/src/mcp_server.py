"""MCP server for SecondBrain: remember/recall/ask from any AI agent.

Connects to SecondBrain API over HTTP — works globally from any repo.

Usage:
  claude mcp add --global secondbrain -- python /path/to/src/mcp_server.py
"""

import asyncio
import json
import logging
import os
import sys

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

logger = logging.getLogger(__name__)

API_URL = os.getenv("SECONDBRAIN_API_URL", "https://memory.atom8.site")
API_KEY = os.getenv("SECONDBRAIN_API_KEY", "")
_MAX_RETRIES = 2
_RETRY_DELAY = 2.0

server = Server("secondbrain")
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=API_URL,
            headers={"X-Api-Key": API_KEY},
            timeout=120.0,
        )
    return _client


async def _api_call(method: str, path: str, **kwargs) -> httpx.Response:
    """HTTP call with retry on 5xx and connection errors."""
    client = _get_client()
    last_exc = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            if method == "GET":
                resp = await client.get(path, **kwargs)
            else:
                resp = await client.post(path, **kwargs)
            if resp.status_code < 500:
                return resp
            last_exc = httpx.HTTPStatusError(
                f"Server error: {resp.status_code}", request=resp.request, response=resp)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            last_exc = e
        if attempt < _MAX_RETRIES:
            await asyncio.sleep(_RETRY_DELAY * (attempt + 1))
            logger.warning("MCP retry %d/%d for %s %s", attempt + 1, _MAX_RETRIES, method, path)
    raise last_exc


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="remember",
            description="Save a fact, decision, or knowledge to SecondBrain. "
                        "Use for important information worth remembering long-term: "
                        "decisions with reasons, project facts, principles, insights.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The knowledge to remember"
                    },
                    "source": {
                        "type": "string",
                        "description": "Where this came from (e.g. 'conversation', 'project-x')",
                        "default": "mcp"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="recall",
            description="Search SecondBrain knowledge graph. Returns relevant context, "
                        "entities, and relationships. Use to find what you know about a topic.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (question or topic)"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["mix", "local", "global"],
                        "description": "mix (default), local (entity-focused), global (broad)",
                        "default": "mix"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="ask",
            description="Ask SecondBrain a question and get an LLM-synthesized answer "
                        "from the knowledge graph. Use for complex questions needing multi-hop reasoning.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to answer"
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="brain_stats",
            description="Get SecondBrain statistics: notes, entities, relations count.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    try:
        if name == "remember":
            text = arguments["text"]
            source = arguments.get("source", "mcp")
            resp = await _api_call("POST", "/add", json={"text": text, "source": source})
            if resp.status_code == 200:
                data = resp.json()
                return [TextContent(type="text", text=f"Saved to SecondBrain: {data.get('path', 'ok')}")]
            elif resp.status_code == 422:
                detail = "quality gate"
                try:
                    detail = resp.json().get("detail", detail)
                except Exception:
                    pass
                return [TextContent(type="text", text=f"Rejected: {detail}")]
            return [TextContent(type="text", text=f"Error: {resp.status_code} {resp.text}")]

        elif name == "recall":
            query = arguments["query"]
            mode = arguments.get("mode", "mix")
            resp = await _api_call("POST", "/search", json={"query": query, "mode": mode})
            if resp.status_code == 200:
                data = resp.json()
                context = data.get("context", "")
                if isinstance(context, dict):
                    context = json.dumps(context, ensure_ascii=False, indent=2)
                return [TextContent(type="text", text=str(context) or "No results found.")]
            return [TextContent(type="text", text=f"Error: {resp.status_code}")]

        elif name == "ask":
            question = arguments["question"]
            resp = await _api_call("POST", "/ask", json={"question": question})
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "No answer.")
                sources = data.get("sources", [])
                if sources:
                    answer += "\n\nSources: " + ", ".join(str(s) for s in sources[:5])
                return [TextContent(type="text", text=answer)]
            return [TextContent(type="text", text=f"Error: {resp.status_code}")]

        elif name == "brain_stats":
            resp = await _api_call("GET", "/stats")
            if resp.status_code == 200:
                data = resp.json()
                lines = [
                    f"Notes: {data.get('total_notes', '?')}",
                    f"Entities: {data.get('entities', '?')}",
                    f"Relations: {data.get('relations', '?')}",
                ]
                by_type = data.get("notes_by_type")
                if by_type and isinstance(by_type, dict):
                    lines.append("By type: " + ", ".join(f"{k}={v}" for k, v in by_type.items()))
                return [TextContent(type="text", text=", ".join(lines))]
            return [TextContent(type="text", text=f"Error: {resp.status_code}")]

        raise ValueError(f"Unknown tool: {name}")

    except (httpx.ConnectError, httpx.TimeoutException) as e:
        return [TextContent(type="text", text=f"Engine unavailable (retries exhausted): {e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
