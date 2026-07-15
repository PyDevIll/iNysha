"""Tavily web search and browse tools for MASTERMIND v2.

Requires: pip install tavily-python
API key: https://tavily.com → TAVILY_API_KEY env var
"""

import os
from typing import Optional
from loguru import logger

try:
    from tavily import AsyncTavilyClient
    HAS_TAVILY = True
except ImportError:
    HAS_TAVILY = False


async def _get_client() -> Optional[AsyncTavilyClient]:
    """Get configured AsyncTavilyClient or None if unavailable."""
    if not HAS_TAVILY:
        return None
    api_key = os.environ.get("TAVILY_API_KEY", "")
    # Tavily allows keyless mode for search/extract with rate limits
    return AsyncTavilyClient(api_key=api_key) if api_key else AsyncTavilyClient()


async def tavily_search(
    query: str,
    search_depth: str = "basic",
    max_results: int = 5,
    include_answer: bool = False,
    topic: str = "general",
    include_domains: str = "",
    exclude_domains: str = "",
    time_range: str = "",
    country: str = "",
    include_raw_content: bool = False,
) -> str:
    """Search the web using Tavily Search API.

    Args:
        query: Search query phrase.
        search_depth: 'basic' (1 credit) or 'advanced' (2 credits, deeper).
        max_results: Maximum results to return (0-20).
        include_answer: Include LLM-generated answer summary.
        topic: Category — 'general', 'news', or 'finance'.
        include_domains: Comma-separated domains to restrict search.
        exclude_domains: Comma-separated domains to exclude.
        time_range: 'day', 'week', 'month', or 'year'.
        country: Country code for region-specific results.
        include_raw_content: Include cleaned raw HTML/markdown of pages.
    """
    if not HAS_TAVILY:
        return "Error: tavily-python not installed. Run: pip install tavily-python"

    try:
        client = await _get_client()
        if client is None:
            return "Error: could not create Tavily client"

        params = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max(1, min(max_results, 20)),
            "include_answer": include_answer,
            "topic": topic,
            "include_raw_content": include_raw_content,
        }

        if include_domains:
            params["include_domains"] = [d.strip() for d in include_domains.split(",") if d.strip()]
        if exclude_domains:
            params["exclude_domains"] = [d.strip() for d in exclude_domains.split(",") if d.strip()]
        if time_range:
            params["time_range"] = time_range
        if country:
            params["country"] = country

        response = await client.search(**params)

        out = []
        # LLM-generated answer
        if include_answer and response.get("answer"):
            out.append(f"**Answer:** {response['answer']}\n")

        # Search results
        results = response.get("results", [])
        if results:
            out.append(f"**{len(results)} result(s) for:** _{query}_")
            for r in results:
                title = r.get("title", "Untitled")
                content = r.get("content", "")
                url = r.get("url", "")
                score = r.get("score", 0)
                published = r.get("published_date", "")
                line = f"\n### [{score:.2f}] {title}\n{content}"
                if published:
                    line += f"\n*Published: {published}*"
                line += f"\n{url}"
                if include_raw_content and r.get("raw_content"):
                    raw = r["raw_content"][:2000]
                    line += f"\n<details><summary>Raw Content</summary>\n\n{raw}\n</details>"
                out.append(line)
        else:
            out.append(f"No results found for: _{query}_")

        # Images
        images = response.get("images", [])
        if images:
            out.append(f"\n**Images ({len(images)}):**")
            for img in images[:6]:
                out.append(f"- ![{img.get('description', '')}]({img.get('url', '')})")

        out.append(f"\n*Response time: {response.get('response_time', '?')}s*")
        return "\n".join(out)

    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return f"Tavily search error: {e}"


async def tavily_browse(
    url: str,
    extract_depth: str = "basic",
    content_format: str = "markdown",
    query: str = "",
    chunks_per_source: int = 3,
    include_images: bool = False,
) -> str:
    """Browse and extract clean content from a URL using Tavily Extract.

    Args:
        url: URL to extract content from. Can be comma-separated for up to 20 URLs.
        extract_depth: 'basic' (1 credit per 5 URLs) or 'advanced' (2 credits per 5 URLs).
        content_format: 'markdown' or 'text'.
        query: Optional query to rank/reorder extracted chunks by relevance.
        chunks_per_source: Number of relevant chunks per source (1-5, requires query).
        include_images: Include extracted images.
    """
    if not HAS_TAVILY:
        return "Error: tavily-python not installed. Run: pip install tavily-python"

    try:
        client = await _get_client()
        if client is None:
            return "Error: could not create Tavily client"

        # Support comma-separated URLs for batch extraction
        urls = [u.strip() for u in url.split(",") if u.strip()]

        params = {
            "urls": urls if len(urls) > 1 else url,
            "extract_depth": extract_depth,
            "format": content_format,
            "include_images": include_images,
        }
        if query:
            params["query"] = query
            params["chunks_per_source"] = max(1, min(chunks_per_source, 5))

        response = await client.extract(**params)

        out = []
        results = response.get("results", [])
        failed = response.get("failed_results", [])

        if results:
            for i, r in enumerate(results):
                raw = r.get("raw_content", "")
                page_url = r.get("url", f"URL #{i+1}")
                out.append(f"## {page_url}\n")
                # Truncate for reasonable context window
                out.append(raw[:12000])
                if len(raw) > 12000:
                    out.append(f"\n\n...*truncated ({len(raw)} chars total)*")
                # Images within this extraction
                imgs = r.get("images", [])
                if imgs:
                    out.append(f"\n\n**Images:** {', '.join(img.get('url', '') for img in imgs[:4])}")
                out.append("")

        if failed:
            out.append(f"**Failed URLs:**")
            for f in failed:
                out.append(f"- {f.get('url', '?')}: {f.get('error', 'unknown')}")

        if not results and not failed:
            out.append("No content extracted.")

        out.append(f"\n*Response time: {response.get('response_time', '?')}s*")
        return "\n".join(out)

    except Exception as e:
        logger.error(f"Tavily browse error: {e}")
        return f"Tavily browse error: {e}"


TOOL_DEFINITIONS = [
    ("tavily_search", tavily_search, "Search the web using Tavily Search API", {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query phrase"},
            "search_depth": {"type": "string", "description": "'basic' (fast) or 'advanced' (deeper, 2 credits)"},
            "max_results": {"type": "integer", "description": "Maximum results 0-20 (default 5)"},
            "include_answer": {"type": "boolean", "description": "Include LLM-generated answer summary"},
            "topic": {"type": "string", "description": "Category: 'general', 'news', or 'finance'"},
            "include_domains": {"type": "string", "description": "Comma-separated domains to restrict to"},
            "exclude_domains": {"type": "string", "description": "Comma-separated domains to exclude"},
            "time_range": {"type": "string", "description": "Time filter: 'day', 'week', 'month', 'year'"},
            "country": {"type": "string", "description": "Country code for regional results"},
            "include_raw_content": {"type": "boolean", "description": "Include cleaned raw page content"},
        },
        "required": ["query"],
    }),
    ("tavily_browse", tavily_browse, "Browse and extract clean content from URLs using Tavily Extract", {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to extract content from (comma-separated for up to 20)"},
            "extract_depth": {"type": "string", "description": "'basic' (1 credit/5 URLs) or 'advanced' (2 credits/5 URLs)"},
            "content_format": {"type": "string", "description": "'markdown' or 'text'"},
            "query": {"type": "string", "description": "Optional query to filter relevant chunks"},
            "chunks_per_source": {"type": "integer", "description": "Relevant chunks per source 1-5 (needs query)"},
            "include_images": {"type": "boolean", "description": "Include extracted images"},
        },
        "required": ["url"],
    }),
]


def register_all(registry):
    for name, func, desc, params in TOOL_DEFINITIONS:
        registry.register_function(func, name, desc, params)
    logger.info(f"Registered {len(TOOL_DEFINITIONS)} Tavily tools")
