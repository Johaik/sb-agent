import sys
import os
import argparse
from typing import List

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db.database import SessionLocal, engine, Base
from src.db.models import ResearchReport, ResearchChunk
from src.db.vector import save_chunks, search_similar_chunks
from src.llm.factory import get_llm_provider
from src.tools.rag_tool import RAGSearchTool
import uuid

def setup_db():
    Base.metadata.create_all(bind=engine)

def seed_data(job_id: str, content_list: List[str]):
    print(f"Seeding RAG data for job {job_id}...")
    llm = get_llm_provider("bedrock")
    db = SessionLocal()
    
    chunks_data = []
    for content in content_list:
        print(f"Embedding: {content[:50]}...")
        embedding = llm.get_embedding(content)
        chunks_data.append({"content": content, "embedding": embedding})
    
    save_chunks(db, job_id, chunks_data)
    db.close()
    print("Seeding complete.")

def test_search(query: str):
    print(f"\nTesting RAG Search for: '{query}'")
    llm = get_llm_provider("bedrock")
    tool = RAGSearchTool(llm)
    
    result = tool.run(query=query)
    print("\n--- Result ---")
    print(result)
    print("--------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG Tool")
    parser.add_argument("--seed", action="store_true", help="Seed dummy data")
    parser.add_argument("--query", type=str, help="Query to search")
    args = parser.parse_args()
    
    # Ensure tables exist
    setup_db()
    
    # Use a fixed dummy job ID for testing
    TEST_JOB_ID = "00000000-0000-0000-0000-000000000001"
    
    # Create dummy report entry if needed for FK constraint
    db = SessionLocal()
    if not db.query(ResearchReport).filter_by(id=TEST_JOB_ID).first():
        print("Creating dummy report parent...")
        report = ResearchReport(id=TEST_JOB_ID, idea="RAG Test", status="completed")
        db.add(report)
        db.commit()
    db.close()

    if args.seed:
        dummy_data = [
            "Sidekiq Pro offers a reliable way to handle batch processing using the Batches feature. It allows callbacks on success.",
            "To configure Redis for Sidekiq, always set the network timeout slightly higher than the Sidekiq timeout.",
            "The best way to scale Sidekiq is by splitting queues into different processes based on priority: critical, default, low.",
            "Using `sidekiq_options retry: false` is recommended for jobs that are not idempotent or handle payment processing."
        ]
        seed_data(TEST_JOB_ID, dummy_data)
    
    if args.query:
        test_search(args.query)
    elif not args.seed:
        print("Please provide --seed to add data or --query 'text' to search.")

