from typing import Dict, Any
from strands.tools import tool
from ..db.database import SessionLocal
from ..db.vector import search_similar_chunks
from ..llm.factory import get_llm_provider

@tool
def rag_search(query: str) -> str:
    """Search the internal research database for relevant information.
    
    Args:
        query: The search query to find relevant research chunks.
        
    Returns:
        String containing relevant information from the internal database.
    """
    # Generate embedding using Bedrock
    try:
        llm = get_llm_provider("bedrock")
        query_embedding = llm.get_embedding(query)
    except Exception as e:
        return f"Error generating embedding: {e}"
    
    # Search DB
    db = SessionLocal()
    try:
        results = search_similar_chunks(db, query_embedding, limit=3)
        if not results:
            return "[RAG] No relevant information found in the internal database."
        
        formatted_results = "\n\n".join([f"Content: {r.content}" for r in results])
        return f"[RAG] Found the following relevant info:\n{formatted_results}"
    except Exception as e:
        return f"Error searching RAG: {e}"
    finally:
        db.close()
