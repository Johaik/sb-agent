from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.db.database import get_db, engine, Base
from src.db.models import ResearchReport
from src.worker.tasks import start_research_chain
import uuid

# Enable pgvector extension
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Research Agent API")

class ResearchRequest(BaseModel):
    idea: str

class ResearchResponse(BaseModel):
    job_id: str
    status: str

@app.post("/research", response_model=ResearchResponse)
def create_research(request: ResearchRequest, db: Session = Depends(get_db)):
    # Create DB entry
    job_id = uuid.uuid4()
    report = ResearchReport(id=job_id, idea=request.idea, status="pending")
    db.add(report)
    db.commit()
    
    # Trigger Worker
    start_research_chain(request.idea, str(job_id))
    
    return {"job_id": str(job_id), "status": "pending"}

@app.get("/research/{job_id}")
def get_research(job_id: str, db: Session = Depends(get_db)):
    report = db.query(ResearchReport).filter(ResearchReport.id == job_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Research job not found")
    
    return {
        "job_id": str(report.id),
        "idea": report.idea,
        "status": report.status,
        "description": report.description,
        "report": report.report,
        "created_at": report.created_at,
        "updated_at": report.updated_at
    }
