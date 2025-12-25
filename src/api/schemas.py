from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID

class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok", "degraded"])
    details: Dict[str, str] = Field(default_factory=dict)

class ResearchRequest(BaseModel):
    idea: str = Field(..., min_length=5, examples=["Future of solid state batteries"])
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "idea": "Market analysis of autonomous farming equipment in 2025"
            }
        }
    )

class ResearchJobStatus(BaseModel):
    job_id: UUID
    status: str = Field(..., examples=["pending", "processing", "completed", "failed"])
    progress_percent: int = Field(0, ge=0, le=100)
    current_phase: str = Field("queued", examples=["enriching", "planning", "researching", "reporting"])
    created_at: datetime
    updated_at: Optional[datetime] = None
    error: Optional[str] = None

class ResearchResult(ResearchJobStatus):
    description: Optional[str] = None
    report: Optional[Dict[str, Any]] = None

