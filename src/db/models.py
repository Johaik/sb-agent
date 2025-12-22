import uuid
from sqlalchemy import Column, String, JSON, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from .database import Base

class ResearchReport(Base):
    __tablename__ = "research_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    idea = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    report = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ResearchChunk(Base):
    __tablename__ = "research_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("research_reports.id"), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1024))  # Assuming Titan Embed Text v2 (1024 dim)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ResearchTask(Base):
    __tablename__ = "research_tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("research_reports.id"), nullable=False)
    title = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending, researching, review, approved, rejected, failed
    result = Column(Text, nullable=True)
    feedback = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

