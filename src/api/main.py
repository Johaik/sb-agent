from fastapi import FastAPI, Depends, HTTPException, Header, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from sqlalchemy import text
import structlog
import redis
import uuid
from typing import Optional

from src.db.database import get_db, engine, Base
from src.db.models import ResearchReport, ResearchTask
from src.worker.tasks import start_research_chain
from src.config import Config
from src.core.logging import setup_logging
from src.api.schemas import ResearchRequest, ResearchJobStatus, ResearchResult, HealthResponse

# Initialize Logging
setup_logging()
logger = structlog.get_logger()

# Enable pgvector extension
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Research Agent API", version="1.0.0")
redis_client = redis.from_url(Config.REDIS_URL)
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Security Dependency
async def verify_api_key(api_key: str = Security(api_key_header)):
    if not Config.API_AUTH_ENABLED:
        return None
    
    if not api_key:
         raise HTTPException(status_code=403, detail="Missing API Key")

    # Handle Bearer token format if present
    token = api_key.replace("Bearer ", "") if api_key.startswith("Bearer ") else api_key
    
    if token != Config.API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return token

# Helper: Calculate Progress
def calculate_progress(job_id: str, db: Session, main_status: str) -> tuple[int, str]:
    """Derives % complete and current phase."""
    if main_status == "completed":
        return 100, "reporting"
    if main_status == "failed":
        return 0, "failed"
    if main_status == "pending":
        return 0, "enriching"
    
    # Check tasks
    tasks = db.query(ResearchTask).filter(ResearchTask.job_id == job_id).all()
    if not tasks:
        # If processing but no tasks yet, we are in planning/enriching
        return 10, "planning"
    
    total = len(tasks)
    completed = sum(1 for t in tasks if t.status in ["APPROVED", "REJECTED"])
    
    # 20% reserved for setup, 70% for tasks, 10% for final report
    if total > 0:
        task_progress = (completed / total) * 70
    else:
        task_progress = 0

    current_progress = 20 + int(task_progress)
    
    phase = "researching"
    if completed == total and total > 0:
        phase = "reporting"
        current_progress = 90
        
    return min(current_progress, 99), phase

@app.get("/health", response_model=HealthResponse)
def liveness_check():
    return {"status": "ok", "details": {"version": "1.0.0"}}

@app.get("/ready", response_model=HealthResponse)
def readiness_check(db: Session = Depends(get_db)):
    status = "ok"
    details = {}
    
    # Check DB
    try:
        db.execute(text("SELECT 1"))
        details["database"] = "connected"
    except Exception as e:
        status = "degraded"
        details["database"] = str(e)
        logger.error("readiness_db_failed", error=str(e))

    # Check Redis
    try:
        redis_client.ping()
        details["redis"] = "connected"
    except Exception as e:
        status = "degraded"
        details["redis"] = str(e)
        logger.error("readiness_redis_failed", error=str(e))

    if status != "ok":
         raise HTTPException(status_code=503, detail=details)
         
    return {"status": status, "details": details}

@app.post("/research", response_model=ResearchJobStatus, dependencies=[Depends(verify_api_key)])
def create_research(
    request: ResearchRequest, 
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    db: Session = Depends(get_db)
):
    # Idempotency Check
    if idempotency_key:
        cached_job_id = redis_client.get(f"idempotency:{idempotency_key}")
        if cached_job_id:
            job_id_str = cached_job_id.decode()
            logger.info("idempotency_hit", key=idempotency_key, job_id=job_id_str)
            return get_research_status_logic(job_id_str, db)

    # Create Job
    job_id = uuid.uuid4()
    report = ResearchReport(id=job_id, idea=request.idea, status="pending")
    db.add(report)
    db.commit()

    # Cache Idempotency Key (Expire in 24h)
    if idempotency_key:
        redis_client.setex(f"idempotency:{idempotency_key}", 86400, str(job_id))

    # Trigger Async Worker
    start_research_chain.delay(request.idea, str(job_id))
    
    logger.info("job_created", job_id=str(job_id), idea_preview=request.idea[:50])
    
    return ResearchJobStatus(
        job_id=job_id,
        status="pending",
        progress_percent=0,
        current_phase="queued",
        created_at=report.created_at
    )

def get_research_status_logic(job_id: str, db: Session) -> ResearchJobStatus:
    report = db.query(ResearchReport).filter(ResearchReport.id == job_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Research job not found")
        
    progress, phase = calculate_progress(str(report.id), db, report.status)
    
    return ResearchJobStatus(
        job_id=report.id,
        status=report.status,
        progress_percent=progress,
        current_phase=phase,
        created_at=report.created_at,
        updated_at=report.updated_at
    )

@app.get("/research/{job_id}", response_model=ResearchResult, dependencies=[Depends(verify_api_key)])
def get_research(job_id: str, db: Session = Depends(get_db)):
    report = db.query(ResearchReport).filter(ResearchReport.id == job_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Research job not found")
    
    progress, phase = calculate_progress(str(report.id), db, report.status)
    
    return ResearchResult(
        job_id=report.id,
        status=report.status,
        progress_percent=progress,
        current_phase=phase,
        created_at=report.created_at,
        updated_at=report.updated_at,
        description=report.description,
        report=report.report
    )
