"""Search utilities powered by Tavily."""

import os
from typing import Any, Dict, List, Optional

from tavily import TavilyClient
from langchain_core.tools import tool


_tavily_client: Optional[TavilyClient] = None


def _get_tavily_client() -> TavilyClient:
    """Lazy-initialize a Tavily client instance."""
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY not set. Please add it to your environment/.env file.")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


def search_web(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Fetch web results for a query using Tavily."""
    try:
        client = _get_tavily_client()
    except RuntimeError as exc:
        print(f"Search disabled: {exc}")
        return []

    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=num_results,
            include_images=False,
            include_answer=False,
        )
    except Exception as exc:
        print(f"Tavily search error: {exc}")
        return []

    results: List[Dict[str, Any]] = []
    for item in response.get("results", [])[:num_results]:
        title = item.get("title") or ""
        url = item.get("url") or ""
        snippet = item.get("content") or ""
        if not title or not url:
            continue

        metadata: Dict[str, Any] = {
            "title": title,
            "url": url,
            "snippet": snippet,
        }

        if item.get("score") is not None:
            metadata["score"] = float(item["score"])

        results.append(metadata)

    return results


@tool("tavily_web_search")
def tavily_search_tool(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Search the public web for current information using Tavily."""
    return search_web(query=query, num_results=num_results)
