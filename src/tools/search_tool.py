"""
src/tools/search_tool.py
─────────────────────────
SearchTool — LangChain BaseTool wrapping DuckDuckGo (or stub).
"""

from __future__ import annotations

import logging
from typing import Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger("multimodal_agent.tools.search")


class SearchInput(BaseModel):
    query: str = Field(description="The search query to look up on the web.")


class SearchTool(BaseTool):
    """
    Web search tool using DuckDuckGo.
    Falls back to stub results if duckduckgo-search is not installed.
    """

    name: str = "web_search"
    description: str = (
        "Search the web for up-to-date information on a topic. "
        "Input should be a concise search query string."
    )
    args_schema: Type[BaseModel] = SearchInput
    use_stub: bool = False
    max_results: int = 5

    def _run(self, query: str) -> str:
        if self.use_stub:
            return self._stub_search(query)
        try:
            from duckduckgo_search import DDGS  # type: ignore
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
            if not results:
                return f"No results found for: {query!r}"
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(
                    f"[{i}] {r.get('title', '')}\n"
                    f"    {r.get('href', '')}\n"
                    f"    {r.get('body', '')[:200]}"
                )
            return "\n\n".join(lines)
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}. Using stub.")
            return self._stub_search(query)

    def _stub_search(self, query: str) -> str:
        return (
            f"[STUB SEARCH] Results for '{query}':\n"
            f"[1] Understanding {query} — A comprehensive overview\n"
            f"    https://example.com/article-1\n"
            f"    This article covers the fundamentals of {query} including "
            f"core concepts, practical applications, and recent developments.\n\n"
            f"[2] {query} in Practice: Tutorial and Examples\n"
            f"    https://example.com/article-2\n"
            f"    Step-by-step guide to implementing {query} in real-world scenarios.\n\n"
            f"[3] Latest Research on {query}\n"
            f"    https://arxiv.org/search/?query={query.replace(' ', '+')}\n"
            f"    Recent peer-reviewed papers and preprints on {query}."
        )

    async def _arun(self, query: str) -> str:
        return self._run(query)

    @classmethod
    def from_config(cls, cfg: dict) -> "SearchTool":
        agent_cfg = cfg.get("agent", {})
        return cls(
            use_stub=not agent_cfg.get("enable_search_tool", True),
        )
