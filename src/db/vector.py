from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timedelta
from typing import Optional
from .models import ResearchChunk
import json

def search_similar_chunks(db: Session, query_embedding: list, limit: int = 5, max_age_days: Optional[int] = None):
    """
    Search for similar chunks using cosine similarity.
    
    Args:
        db: Database session
        query_embedding: Vector embedding for similarity search
        limit: Maximum number of results to return
        max_age_days: Optional maximum age in days. If specified, only chunks
                      created within this many days will be returned.
    
    Returns:
        List of ResearchChunk objects ordered by similarity
    """
    query = db.query(ResearchChunk)
    
    # Apply age filter if specified
    if max_age_days is not None:
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        query = query.filter(ResearchChunk.created_at >= cutoff_date)
    
    results = query.order_by(
        ResearchChunk.embedding.cosine_distance(query_embedding)
    ).limit(limit).all()
    
    return results

def save_chunks(db: Session, job_id: str, report_data: any, llm_provider):
    """
    Save report chunks with embeddings.
    report_data: dict or str
    llm_provider: LLMProvider with get_embedding method
    """
    chunks = []
    
    # 1. Flatten content
    text_content = ""
    if isinstance(report_data, dict):
        # Extract summary and details
        summary = report_data.get("summary", "")
        details = report_data.get("details", {})
        
        text_content += f"Summary:\n{summary}\n\n"
        if isinstance(details, dict):
            for section, content in details.items():
                text_content += f"Section: {section}\n{content}\n\n"
        elif isinstance(details, list):
             for item in details:
                 text_content += f"{item}\n\n"
        else:
            text_content += str(details)
    else:
        text_content = str(report_data)
        
    # 2. Split into chunks (simple naive split by paragraphs or chars)
    # Ideally use a text splitter
    raw_chunks = [c.strip() for c in text_content.split('\n\n') if c.strip()]
    
    # 3. Embed and Create Objects
    for chunk_text in raw_chunks:
        # Skip very short chunks
        if len(chunk_text) < 50:
            continue
            
        try:
            embedding = llm_provider.get_embedding(chunk_text)
            
            chunk_obj = ResearchChunk(
                job_id=job_id,
                content=chunk_text,
                embedding=embedding
            )
            chunks.append(chunk_obj)
        except Exception as e:
            print(f"Failed to embed chunk: {e}")
            continue
    
    if chunks:
        db.add_all(chunks)
        db.commit()
