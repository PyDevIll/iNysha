"""Tavily web search, extract, map and research tools for MASTERMIND v2.

Requires: pip install tavily-python
API key: https://tavily.com -> TAVILY_API_KEY env var

SDK capabilities used here (tavily-python >= 0.7):
  - search:  fast / ultra-fast / basic / advanced depths, days & start_date/end_date,
             exact_match, include_answer='basic'|'advanced', include_raw_content='markdown',
             auto_parameters, language filter
  - extract: query+chunks_per_source ranking, own content cap (SDK silently stops at 20k)
  - map:     site structure discovery (URL list)
  - research: async deep-research task (mini/pro), polled to completion
"""

import asyncio
from typing import Optional

from loguru import logger

from server import config

try:
    from tavily import AsyncTavilyClient
    HAS_TAVILY = True
except ImportError:
    HAS_TAVILY = False

# Tavily hard-caps each extracted page at ~20k chars; asking for less is pointless.
MAX_PAGE_CHARS = 10000
MAX_PAGE_CHARS_QUERY = 6000  # when query-ranking is used, chunks are smaller


async def _get_client() -> Optional["AsyncTavilyClient"]:
    """Get configured AsyncTavilyClient or None if unavailable."""
    if not HAS_TAVILY:
        return None
    api_key = config.TAVILY_API_KEY
    # Tavily allows keyless mode for search/extract with rate limits
    return AsyncTavilyClient(api_key=api_key) if api_key else AsyncTavilyClient()


def _split(value: str) -> list[str]:
    return [d.strip() for d in (value or "").split(",") if d.strip()]


def _answer_value(include_answer: bool, depth: str) -> Optional[str]:
    """Map boolean flag to SDK's bool|'basic'|'advanced' answer parameter."""
    if not include_answer:
        return None
    return "advanced" if depth == "advanced" else True


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
    days: int = 0,
    start_date: str = "",
    end_date: str = "",
    exact_match: bool = False,
    auto_parameters: bool = False,
    language: str = "",
    snippet_chars: int = 0,
) -> str:
    """Search the web using Tavily Search API.

    Args:
        query: Search query phrase.
        search_depth: 'basic', 'fast', 'ultra-fast' or 'advanced' (deeper, 2 credits).
        max_results: Maximum results to return (0-20).
        include_answer: Include LLM-generated answer summary.
        topic: Category — 'general', 'news', or 'finance'.
        include_domains: Comma-separated domains to restrict search.
        exclude_domains: Comma-separated domains to exclude.
        time_range: 'day', 'week', 'month', or 'year'.
        country: Country code for region-specific results.
        include_raw_content: Include cleaned raw page content (markdown).
        days: Only results from the last N days (news topic mostly).
        start_date: 'YYYY-MM-DD' lower bound for results.
        end_date: 'YYYY-MM-DD' upper bound for results.
        exact_match: Only return pages containing the exact quoted phrase.
        auto_parameters: Let Tavily pick depth/topic itself.
        language: Result language filter (e.g. 'russian').
        snippet_chars: Truncate each snippet to N chars (0 = full).
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
            "topic": topic,
        }

        answer = _answer_value(include_answer, search_depth)
        if answer is not None:
            params["include_answer"] = answer
        if include_raw_content:
            params["include_raw_content"] = "markdown"

        if include_domains:
            params["include_domains"] = _split(include_domains)
        if exclude_domains:
            params["exclude_domains"] = _split(exclude_domains)
        if time_range:
            params["time_range"] = time_range
        if country:
            params["country"] = country
        if days and days > 0:
            params["days"] = days
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if exact_match:
            params["exact_match"] = True
        if auto_parameters:
            params["auto_parameters"] = True
        if language:
            params["language"] = language
        if include_raw_content:
            # raw pages are the reason for context bloat - cap them here
            params["max_results"] = min(params["max_results"], 5)

        response = await client.search(**params)

        out = []
        if include_answer and response.get("answer"):
            out.append(f"**Answer:** {response['answer']}\n")

        results = response.get("results", [])
        if results:
            out.append(f"**{len(results)} result(s) for:** _{query}_")
            for r in results:
                title = r.get("title", "Untitled")
                content = r.get("content", "") or ""
                if snippet_chars and snippet_chars > 0:
                    content = content[:snippet_chars]
                url = r.get("url", "")
                score = r.get("score", 0) or 0
                published = r.get("published_date", "")
                line = f"\n### [{score:.2f}] {title}\n{content}"
                if published:
                    line += f"\n*Published: {published}*"
                line += f"\n{url}"
                raw = r.get("raw_content")
                if include_raw_content and raw:
                    line += (
                        f"\n<details><summary>Raw Content</summary>\n\n"
                        f"{raw[:MAX_PAGE_CHARS]}\n</details>"
                    )
                out.append(line)
        else:
            out.append(f"No results found for: _{query}_")

        images = response.get("images", [])
        if images:
            out.append(f"\n**Images ({len(images)}):**")
            for img in images[:6]:
                out.append(f"- ![{img.get('description', '')}]({img.get('url', '')})")

        response_time = (
            response.get("response_time")
            or (response.get("usage") or {}).get("response_time")
            or "?"
        )
        out.append(f"\n*Response time: {response_time}s*")
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
    max_chars: int = MAX_PAGE_CHARS,
) -> str:
    """Browse and extract clean content from a URL using Tavily Extract.

    Args:
        url: URL to extract content from. Can be comma-separated for up to 20 URLs.
        extract_depth: 'basic' (1 credit per 5 URLs) or 'advanced' (2 credits per 5 URLs).
        content_format: 'markdown' or 'text'.
        query: Optional query to rank/reorder extracted chunks by relevance.
        chunks_per_source: Number of relevant chunks per source (1-5, requires query).
        include_images: Include extracted images.
        max_chars: Per-page character cap for the returned content.
    """
    if not HAS_TAVILY:
        return "Error: tavily-python not installed. Run: pip install tavily-python"

    try:
        client = await _get_client()
        if client is None:
            return "Error: could not create Tavily client"

        urls = _split(url)
        if not urls:
            return "Error: empty url"

        params = {
            "urls": urls if len(urls) > 1 else urls[0],
            "extract_depth": extract_depth,
            "format": content_format,
            "include_images": include_images,
        }
        limit = MAX_PAGE_CHARS
        if query:
            params["query"] = query
            params["chunks_per_source"] = max(1, min(chunks_per_source, 5))
            limit = min(max_chars, MAX_PAGE_CHARS_QUERY)
        else:
            limit = max_chars if max_chars > 0 else MAX_PAGE_CHARS
            # without a query Tavily returns the whole page - keep it sane
            if len(urls) > 3:
                limit = min(limit, 4000)

        response = await client.extract(**params)

        out = []
        results = response.get("results", [])
        failed = response.get("failed_results", [])

        for i, r in enumerate(results):
            raw = r.get("raw_content", "") or ""
            page_url = r.get("url", f"URL #{i+1}")
            out.append(f"## {page_url}\n")
            out.append(raw[:limit])
            if len(raw) > limit:
                out.append(f"\n\n...*truncated ({len(raw)} chars total)*")
            imgs = r.get("images", [])
            if imgs:
                out.append(
                    f"\n\n**Images:** {', '.join(img.get('url', '') for img in imgs[:4])}"
                )
            out.append("")

        if failed:
            out.append("**Failed URLs:**")
            for f in failed:
                out.append(f"- {f.get('url', '?')}: {f.get('error', 'unknown')}")

        if not results and not failed:
            out.append("No content extracted.")

        response_time = response.get("response_time", "?")
        out.append(f"\n*Response time: {response_time}s*")
        return "\n".join(out)

    except Exception as e:
        logger.error(f"Tavily browse error: {e}")
        return f"Tavily browse error: {e}"


async def tavily_map(
    url: str,
    max_depth: int = 2,
    limit: int = 50,
    instructions: str = "",
    select_paths: str = "",
    exclude_paths: str = "",
    allow_external: bool = False,
) -> str:
    """Discover the URL structure of a website (Tavily Map).

    Args:
        url: Base URL of the site.
        max_depth: How deep to follow links (1-5).
        limit: Maximum number of URLs to return.
        instructions: Natural-language hint about which pages matter.
        select_paths: Comma-separated path regexes to include.
        exclude_paths: Comma-separated path regexes to exclude.
        allow_external: Follow links to other domains.
    """
    if not HAS_TAVILY:
        return "Error: tavily-python not installed. Run: pip install tavily-python"

    try:
        client = await _get_client()
        if client is None:
            return "Error: could not create Tavily client"

        params = {
            "url": url,
            "max_depth": max(1, min(max_depth, 5)),
            "limit": max(1, min(limit, 200)),
            "allow_external": allow_external,
        }
        if instructions:
            params["instructions"] = instructions
        if select_paths:
            params["select_paths"] = _split(select_paths)
        if exclude_paths:
            params["exclude_paths"] = _split(exclude_paths)

        response = await client.map(**params)
        raw_results = response.get("results", [])
        # map returns a plain list of URL strings (unlike extract/search)
        urls = [
            (r.get("url", "") if isinstance(r, dict) else str(r))
            for r in raw_results
        ]
        urls = [u for u in urls if u]
        if not urls:
            return f"No URLs discovered for {url}"

        base = response.get("base_url") or url
        lines = [f"**{len(urls)} URL(s) on {base}:**"] + [f"- {u}" for u in urls]
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Tavily map error: {e}")
        return f"Tavily map error: {e}"


async def tavily_research(
    input: str,
    model: str = "mini",
    citation_format: str = "numbered",
    poll_seconds: int = 6,
    max_wait_seconds: int = 240,
) -> str:
    """Run a deep-research task and return the final report with sources.

    Args:
        input: Research question or task description.
        model: 'mini' (fast), 'pro' (deeper, more credits) or 'auto'.
        citation_format: 'numbered', 'mla', 'apa' or 'chicago'.
        poll_seconds: Polling interval while the task runs.
        max_wait_seconds: Give up after this many seconds.
    """
    if not HAS_TAVILY:
        return "Error: tavily-python not installed. Run: pip install tavily-python"

    try:
        client = await _get_client()
        if client is None:
            return "Error: could not create Tavily client"

        created = await client.research(
            input=input, model=model, citation_format=citation_format
        )
        request_id = created.get("request_id")
        if not request_id:
            return f"Research task not created: {created}"

        waited = 0
        result = {}
        while waited < max_wait_seconds:
            await asyncio.sleep(poll_seconds)
            waited += poll_seconds
            result = await client.get_research(request_id)
            if (result.get("status") or "").lower() in {"completed", "failed", "error"}:
                break

        status = result.get("status", "unknown")
        content = result.get("content") or ""
        sources = result.get("sources") or []

        out = [f"**Research status:** {status} (waited {waited}s)"]
        if content:
            out.append(content)
        if sources:
            out.append(f"\n**Sources ({len(sources)}):**")
            for i, s in enumerate(sources[:20], 1):
                if isinstance(s, dict):
                    out.append(f"{i}. {s.get('title') or s.get('url', '')} — {s.get('url', '')}")
                else:
                    out.append(f"{i}. {s}")
        if not content and not sources:
            out.append(f"Nothing returned yet; request_id={request_id}")
        return "\n".join(out)

    except Exception as e:
        logger.error(f"Tavily research error: {e}")
        return f"Tavily research error: {e}"


TOOL_DEFINITIONS = [
    ("tavily_search", tavily_search, "Search the web using Tavily Search API", {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query phrase"},
            "search_depth": {"type": "string", "description": "'fast'/'ultra-fast' (cheap), 'basic' (fast) or 'advanced' (deeper, 2 credits)"},
            "max_results": {"type": "integer", "description": "Maximum results 0-20 (default 5)"},
            "include_answer": {"type": "boolean", "description": "Include LLM-generated answer summary"},
            "topic": {"type": "string", "description": "Category: 'general', 'news', or 'finance'"},
            "include_domains": {"type": "string", "description": "Comma-separated domains to restrict to"},
            "exclude_domains": {"type": "string", "description": "Comma-separated domains to exclude"},
            "time_range": {"type": "string", "description": "Time filter: 'day', 'week', 'month', 'year'"},
            "country": {"type": "string", "description": "Country code for regional results"},
            "include_raw_content": {"type": "boolean", "description": "Include cleaned raw page content (markdown, capped)"},
            "days": {"type": "integer", "description": "Only results from the last N days (0 = off)"},
            "start_date": {"type": "string", "description": "Lower date bound YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "Upper date bound YYYY-MM-DD"},
            "exact_match": {"type": "boolean", "description": "Only pages containing the exact quoted phrase"},
            "auto_parameters": {"type": "boolean", "description": "Let Tavily choose depth/topic automatically"},
            "language": {"type": "string", "description": "Language filter, e.g. 'russian'"},
            "snippet_chars": {"type": "integer", "description": "Truncate each snippet to N chars (0 = full)"},
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
            "max_chars": {"type": "integer", "description": "Per-page character cap for returned content"},
        },
        "required": ["url"],
    }),
    ("tavily_map", tavily_map, "Discover the URL structure of a website using Tavily Map", {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Base URL of the site"},
            "max_depth": {"type": "integer", "description": "How deep to follow links (1-5, default 2)"},
            "limit": {"type": "integer", "description": "Max URLs to return (default 50)"},
            "instructions": {"type": "string", "description": "Natural-language hint which pages matter"},
            "select_paths": {"type": "string", "description": "Comma-separated path regexes to include"},
            "exclude_paths": {"type": "string", "description": "Comma-separated path regexes to exclude"},
            "allow_external": {"type": "boolean", "description": "Follow links to other domains"},
        },
        "required": ["url"],
    }),
    ("tavily_research", tavily_research, "Run a deep-research task and return the report with sources", {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Research question or task"},
            "model": {"type": "string", "description": "'mini' (fast), 'pro' (deeper) or 'auto'"},
            "citation_format": {"type": "string", "description": "'numbered', 'mla', 'apa' or 'chicago'"},
            "poll_seconds": {"type": "integer", "description": "Polling interval, default 6"},
            "max_wait_seconds": {"type": "integer", "description": "Give up after N seconds, default 240"},
        },
        "required": ["input"],
    }),
]


def register_all(registry):
    for name, func, desc, params in TOOL_DEFINITIONS:
        registry.register_function(func, name, desc, params)
    logger.info(f"Registered {len(TOOL_DEFINITIONS)} Tavily tools")
