from sqlalchemy.orm import Session
from sqlalchemy import text
from .models import ResearchChunk
import json

def search_similar_chunks(db: Session, query_embedding: list, limit: int = 5):
    """
    Search for similar chunks using cosine similarity.
    """
    results = db.query(ResearchChunk).order_by(
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
