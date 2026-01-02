from typing import Dict, Any, Optional
from datetime import datetime
from strands.tools import tool
from ..db.database import SessionLocal
from ..db.vector import search_similar_chunks
from ..llm.factory import get_llm_provider

@tool
def rag_search(query: str, max_age_days: Optional[int] = None) -> str:
    """Search the internal research database for relevant information.
    
    Args:
        query: The search query to find relevant research chunks.
        max_age_days: Optional maximum age in days for results. Use this for
                      time-sensitive queries (e.g., max_age_days=7 for recent news,
                      max_age_days=30 for monthly data). If not specified, all
                      results are returned with their age metadata.
        
    Returns:
        String containing relevant information from the internal database,
        including age metadata for each result.
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
        results = search_similar_chunks(db, query_embedding, limit=3, max_age_days=max_age_days)
        if not results:
            age_note = f" (within last {max_age_days} days)" if max_age_days else ""
            return f"[RAG] No relevant information found in the internal database{age_note}."
        
        # Format results with age metadata
        formatted_parts = []
        now = datetime.utcnow()
        for idx, r in enumerate(results, 1):
            # Calculate age
            created_at = r.created_at
            if created_at:
                # Handle timezone-aware datetimes
                if created_at.tzinfo is not None:
                    created_at = created_at.replace(tzinfo=None)
                age_delta = now - created_at
                age_days = age_delta.days
                date_str = created_at.strftime("%Y-%m-%d")
                
                if age_days == 0:
                    age_text = "today"
                elif age_days == 1:
                    age_text = "1 day ago"
                else:
                    age_text = f"{age_days} days ago"
                
                header = f"--- Result {idx} (Retrieved: {date_str}, {age_text}) ---"
            else:
                header = f"--- Result {idx} (age unknown) ---"
            
            formatted_parts.append(f"{header}\nContent: {r.content}")
        
        formatted_results = "\n\n".join(formatted_parts)
        return f"[RAG] Found the following relevant info:\n\n{formatted_results}"
    except Exception as e:
        return f"Error searching RAG: {e}"
    finally:
        db.close()
