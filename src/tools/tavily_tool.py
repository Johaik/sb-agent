from typing import Dict, Any
from .base import Tool
from tavily import TavilyClient
from ..config import Config

class TavilyTool(Tool):
    def __init__(self):
        self.client = TavilyClient(api_key=Config.TAVILY_API_KEY)
        
        super().__init__(
            name="tavily_search",
            description="Search the web using Tavily. Best for current events and broad research.",
            parameters={
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"]
            }
        )

    def run(self, invocation_state: Dict[str, Any] = None, **kwargs) -> Any:
        query = kwargs.get('query')
        try:
            return self.client.search(query=query, search_depth="advanced")
        except Exception as e:
            return f"Error performing search: {e}"
