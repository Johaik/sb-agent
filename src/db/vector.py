from sqlalchemy.orm import Session
from sqlalchemy import text
from .models import ResearchChunk
import numpy as np

def search_similar_chunks(db: Session, query_embedding: list, limit: int = 5):
    """
    Search for similar chunks using cosine similarity.
    """
    # Create the vector string representation for pgvector
    # Note: pgvector expects [1,2,3] format
    
    # Using the l2_distance or cosine_distance operator
    # <-> is L2 distance, <=> is cosine distance, <#> is negative inner product
    
    results = db.query(ResearchChunk).order_by(
        ResearchChunk.embedding.cosine_distance(query_embedding)
    ).limit(limit).all()
    
    return results

def save_chunks(db: Session, job_id, chunks_data: list):
    """
    Save multiple chunks with embeddings.
    chunks_data: list of dicts with 'content' and 'embedding'
    """
    chunks = []
    for data in chunks_data:
        chunk = ResearchChunk(
            job_id=job_id,
            content=data["content"],
            embedding=data["embedding"]
        )
        chunks.append(chunk)
    
    db.add_all(chunks)
    db.commit()

