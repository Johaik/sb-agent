from typing import Dict, Any, List, Optional
from strands.tools import tool
from tavily import TavilyClient
from ..config import Config
from ..llm.factory import get_llm_provider

@tool
def tavily_search(
    query: str,
    search_depth: str = "advanced",
    max_results: int = 5,
    include_raw_content: bool = False,
    deep_search: bool = False
) -> Dict[str, Any]:
    """Search the web using Tavily. Best for current events and broad research.
    
    Args:
        query: The search query.
        search_depth: Depth of search. 'advanced' is deeper and more thorough.
        max_results: Number of results to return.
        include_raw_content: Whether to include full page content.
        deep_search: If true, performs multiple searches to get comprehensive results.
    
    Returns:
        Dictionary with 'answer' and 'results' keys containing search results.
    """
    client = TavilyClient(api_key=Config.TAVILY_API_KEY)
    
    try:
        if deep_search:
            return _run_deep_search(client, query, search_depth, max_results, include_raw_content)
        
        # Standard search
        response = client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_raw_content=include_raw_content,
            include_answer=True
        )
        
        return _format_output(response.get("answer", ""), response.get("results", []))
            
    except Exception as e:
        return {"error": f"Error performing search: {e}"}


def _run_deep_search(client: TavilyClient, query: str, search_depth: str, max_results: int, include_raw_content: bool) -> Dict[str, Any]:
    """Generate sub-queries using LLM and perform multiple searches."""
    queries = [query]
    
    # Generate sub-queries using LLM
    try:
        llm = get_llm_provider("bedrock")
        messages = [
            {"role": "system", "content": "You are a research assistant. Generate 3 distinct, specific search queries to comprehensively research the user's topic. Output ONLY the queries, one per line."},
            {"role": "user", "content": query}
        ]
        llm_response = llm.generate(messages)
        content = llm_response.get("content", "")
        queries = [q.strip() for q in content.split('\n') if q.strip()]
        if not queries:
            queries = [query]
    except Exception:
        queries = [query]  # Fallback to original query
        
    # Add original query if not in list
    if query not in queries:
        queries.insert(0, query)
        
    all_results = []
    answer = ""
    
    # Limit total results to avoid overwhelming
    results_per_query = max(2, max_results // len(queries))
    
    for q in queries[:4]:  # Limit to 4 queries max
        try:
            resp = client.search(
                query=q,
                search_depth=search_depth,
                max_results=results_per_query,
                include_raw_content=include_raw_content,
                include_answer=True
            )
            
            if resp.get("answer") and not answer:
                answer = resp.get("answer")
            
            if "results" in resp:
                all_results.extend(resp["results"])
        except Exception:
            continue

    # Deduplicate by URL
    unique_results = []
    seen_urls = set()
    for r in all_results:
        if r["url"] not in seen_urls:
            unique_results.append(r)
            seen_urls.add(r["url"])
            
    # Trim to max_results (maybe a bit more for deep search)
    final_results = unique_results[:max_results * 2] 
    
    return _format_output(answer, final_results)


def _format_output(answer: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format search results into a consistent structure."""
    formatted_results = []
    for r in results:
        content = r.get("raw_content") if r.get("raw_content") else r.get("content")
        # Truncate content if too long to save tokens
        if content and len(content) > 5000:
            content = content[:5000] + "...(truncated)"
            
        formatted_results.append({
            "title": r.get("title"),
            "url": r.get("url"),
            "content": content,
            "score": r.get("score")
        })
        
    return {
        "answer": answer,
        "results": formatted_results
    }
