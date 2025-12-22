from typing import Dict, Any, List
from .base import Tool
from ..db.database import SessionLocal
from ..db.vector import search_similar_chunks
from ..llm.base import LLMProvider

class RAGSearchTool(Tool):
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        super().__init__(
            name="rag_search",
            description="Search the internal research database for relevant information.",
            parameters={
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant research chunks."
                    }
                },
                "required": ["query"]
            }
        )

    def run(self, query: str) -> str:
        # Generate embedding
        try:
            query_embedding = self.llm.get_embedding(query)
        except Exception as e:
            return f"Error generating embedding: {e}"
        
        # Search DB
        db = SessionLocal()
        try:
            results = search_similar_chunks(db, query_embedding, limit=3)
            if not results:
                return "No relevant information found in the internal database."
            
            formatted_results = "\n\n".join([f"Content: {r.content}" for r in results])
            return f"Found the following relevant info:\n{formatted_results}"
        except Exception as e:
            return f"Error searching RAG: {e}"
        finally:
            db.close()

