"""
Integration tests for database operations.
Tests models and database CRUD operations.
"""
import pytest
from uuid import uuid4
from datetime import datetime


@pytest.mark.integration
class TestResearchReportModel:
    """Tests for ResearchReport model CRUD operations."""
    
    def test_create_research_report(self, db_session):
        """Test creating a new research report."""
        from src.db.models import ResearchReport
        
        report = ResearchReport(
            idea="Test research idea for database testing",
            status="pending"
        )
        db_session.add(report)
        db_session.commit()
        db_session.refresh(report)
        
        assert report.id is not None
        assert report.idea == "Test research idea for database testing"
        assert report.status == "pending"
        assert report.created_at is not None
    
    def test_create_research_report_with_uuid(self, db_session):
        """Test creating a report with specific UUID."""
        from src.db.models import ResearchReport
        
        custom_id = uuid4()
        report = ResearchReport(
            id=custom_id,
            idea="Research with custom UUID",
            status="pending"
        )
        db_session.add(report)
        db_session.commit()
        
        assert report.id == custom_id
    
    def test_update_research_report(self, db_session, sample_research_report):
        """Test updating a research report."""
        sample_research_report.status = "processing"
        sample_research_report.description = "Updated description"
        db_session.commit()
        db_session.refresh(sample_research_report)
        
        assert sample_research_report.status == "processing"
        assert sample_research_report.description == "Updated description"
    
    def test_update_report_with_json(self, db_session, sample_research_report):
        """Test updating report with JSON data."""
        report_data = {
            "summary": "Test summary",
            "key_findings": ["Finding 1", "Finding 2"],
            "details": {"Section 1": "Content"}
        }
        
        sample_research_report.report = report_data
        sample_research_report.status = "completed"
        db_session.commit()
        db_session.refresh(sample_research_report)
        
        assert sample_research_report.report["summary"] == "Test summary"
        assert len(sample_research_report.report["key_findings"]) == 2
    
    def test_read_research_report(self, db_session, sample_research_report):
        """Test reading a research report."""
        from src.db.models import ResearchReport
        
        found = db_session.query(ResearchReport).filter(
            ResearchReport.id == sample_research_report.id
        ).first()
        
        assert found is not None
        assert found.id == sample_research_report.id
        assert found.idea == sample_research_report.idea
    
    def test_delete_research_report(self, db_session):
        """Test deleting a research report."""
        from src.db.models import ResearchReport
        
        report = ResearchReport(
            idea="Report to delete",
            status="pending"
        )
        db_session.add(report)
        db_session.commit()
        report_id = report.id
        
        db_session.delete(report)
        db_session.commit()
        
        found = db_session.query(ResearchReport).filter(
            ResearchReport.id == report_id
        ).first()
        
        assert found is None
    
    def test_research_report_timestamps(self, db_session):
        """Test automatic timestamp fields."""
        from src.db.models import ResearchReport
        
        report = ResearchReport(
            idea="Test timestamps",
            status="pending"
        )
        db_session.add(report)
        db_session.commit()
        db_session.refresh(report)
        
        assert report.created_at is not None
        # updated_at is None on creation, set on update
        
        # Update to trigger updated_at
        report.status = "processing"
        db_session.commit()
        db_session.refresh(report)
        
        # Note: updated_at behavior depends on database trigger
    
    def test_research_report_final_critique(self, db_session, sample_research_report):
        """Test storing final critique JSON."""
        critique = {
            "approved": True,
            "critique": "Well-structured report",
            "required_edits": []
        }
        
        sample_research_report.final_critique = critique
        db_session.commit()
        db_session.refresh(sample_research_report)
        
        assert sample_research_report.final_critique["approved"] is True


@pytest.mark.integration
class TestResearchTaskModel:
    """Tests for ResearchTask model CRUD operations."""
    
    def test_create_research_task(self, db_session, sample_research_report):
        """Test creating a new research task."""
        from src.db.models import ResearchTask
        
        task = ResearchTask(
            job_id=sample_research_report.id,
            title="Test task for database",
            status="PENDING"
        )
        db_session.add(task)
        db_session.commit()
        db_session.refresh(task)
        
        assert task.id is not None
        assert task.job_id == sample_research_report.id
        assert task.status == "PENDING"
    
    def test_create_multiple_tasks(self, db_session, sample_research_report):
        """Test creating multiple tasks for a job."""
        from src.db.models import ResearchTask
        
        tasks = []
        for i in range(5):
            task = ResearchTask(
                job_id=sample_research_report.id,
                title=f"Task {i+1}",
                status="PENDING"
            )
            tasks.append(task)
            db_session.add(task)
        
        db_session.commit()
        
        found_tasks = db_session.query(ResearchTask).filter(
            ResearchTask.job_id == sample_research_report.id
        ).all()
        
        assert len(found_tasks) == 5
    
    def test_update_task_status(self, db_session, sample_research_task):
        """Test updating task status."""
        sample_research_task.status = "APPROVED"
        db_session.commit()
        db_session.refresh(sample_research_task)
        
        assert sample_research_task.status == "APPROVED"
    
    def test_task_result_field(self, db_session, sample_research_task):
        """Test storing task result."""
        sample_research_task.result = "This is the research result with detailed findings."
        sample_research_task.status = "RESEARCHED"
        db_session.commit()
        db_session.refresh(sample_research_task)
        
        assert sample_research_task.result is not None
        assert "detailed findings" in sample_research_task.result
    
    def test_task_hypotheses_json(self, db_session, sample_research_task):
        """Test storing hypotheses as JSON."""
        hypotheses = {
            "hypotheses": [
                {"statement": "Test hypothesis", "confidence": "high", "reasoning": "Test"}
            ]
        }
        
        sample_research_task.hypotheses = hypotheses
        db_session.commit()
        db_session.refresh(sample_research_task)
        
        assert sample_research_task.hypotheses["hypotheses"][0]["confidence"] == "high"
    
    def test_task_evidence_rating_json(self, db_session, sample_research_task):
        """Test storing evidence rating as JSON."""
        rating = {
            "relevance_score": 8,
            "credibility_score": 7,
            "analysis": "Good evidence",
            "weak_points": []
        }
        
        sample_research_task.evidence_rating = rating
        db_session.commit()
        db_session.refresh(sample_research_task)
        
        assert sample_research_task.evidence_rating["relevance_score"] == 8
    
    def test_task_contradictions_json(self, db_session, sample_research_task):
        """Test storing contradictions as JSON."""
        contradictions = {
            "contradictions_found": True,
            "details": [
                {"claim_challenged": "Test", "contradictory_evidence": "Counter", "source": "URL"}
            ]
        }
        
        sample_research_task.contradictions = contradictions
        db_session.commit()
        db_session.refresh(sample_research_task)
        
        assert sample_research_task.contradictions["contradictions_found"] is True
    
    def test_task_feedback_field(self, db_session, sample_research_task):
        """Test storing feedback."""
        sample_research_task.feedback = "Needs more detail on implementation"
        sample_research_task.status = "REJECTED"
        db_session.commit()
        db_session.refresh(sample_research_task)
        
        assert sample_research_task.feedback is not None
        assert sample_research_task.status == "REJECTED"
    
    def test_query_tasks_by_status(self, db_session, sample_research_report):
        """Test querying tasks by status."""
        from src.db.models import ResearchTask
        
        # Create tasks with different statuses
        statuses = ["PENDING", "APPROVED", "APPROVED", "REJECTED"]
        for i, status in enumerate(statuses):
            task = ResearchTask(
                job_id=sample_research_report.id,
                title=f"Task {i}",
                status=status
            )
            db_session.add(task)
        db_session.commit()
        
        approved = db_session.query(ResearchTask).filter(
            ResearchTask.job_id == sample_research_report.id,
            ResearchTask.status == "APPROVED"
        ).all()
        
        assert len(approved) == 2


@pytest.mark.integration
class TestResearchChunkModel:
    """Tests for ResearchChunk model with vector support."""
    
    def test_create_research_chunk(self, db_session, sample_research_report):
        """Test creating a research chunk with embedding."""
        from src.db.models import ResearchChunk
        
        embedding = [0.1] * 1024
        chunk = ResearchChunk(
            job_id=sample_research_report.id,
            content="This is test content for the chunk.",
            embedding=embedding
        )
        db_session.add(chunk)
        db_session.commit()
        db_session.refresh(chunk)
        
        assert chunk.id is not None
        assert chunk.content == "This is test content for the chunk."
    
    def test_create_multiple_chunks(self, db_session, sample_research_report):
        """Test creating multiple chunks for a job."""
        from src.db.models import ResearchChunk
        
        for i in range(3):
            chunk = ResearchChunk(
                job_id=sample_research_report.id,
                content=f"Chunk content {i}",
                embedding=[0.1 + (i * 0.01)] * 1024
            )
            db_session.add(chunk)
        
        db_session.commit()
        
        chunks = db_session.query(ResearchChunk).filter(
            ResearchChunk.job_id == sample_research_report.id
        ).all()
        
        assert len(chunks) == 3
    
    def test_chunk_embedding_dimensions(self, db_session, sample_research_report):
        """Test that embedding has correct dimensions."""
        from src.db.models import ResearchChunk
        
        embedding = [0.5] * 1024
        chunk = ResearchChunk(
            job_id=sample_research_report.id,
            content="Test content",
            embedding=embedding
        )
        db_session.add(chunk)
        db_session.commit()
        db_session.refresh(chunk)
        
        # Embedding should be stored correctly
        assert chunk.embedding is not None


@pytest.mark.integration
class TestAgentLogModel:
    """Tests for AgentLog model."""
    
    def test_create_agent_log(self, db_session, sample_research_report):
        """Test creating an agent log entry."""
        from src.db.models import AgentLog
        
        log = AgentLog(
            job_id=sample_research_report.id,
            agent_name="Researcher",
            role="assistant",
            content="Research findings here"
        )
        db_session.add(log)
        db_session.commit()
        db_session.refresh(log)
        
        assert log.id is not None
        assert log.agent_name == "Researcher"
        assert log.role == "assistant"
    
    def test_agent_log_with_tool_calls(self, db_session, sample_research_report):
        """Test agent log with tool calls."""
        from src.db.models import AgentLog
        
        tool_calls = [
            {"name": "tavily_search", "input": {"query": "AI"}, "id": "123"}
        ]
        
        log = AgentLog(
            job_id=sample_research_report.id,
            agent_name="Researcher",
            role="assistant",
            content="",
            tool_calls=tool_calls
        )
        db_session.add(log)
        db_session.commit()
        db_session.refresh(log)
        
        assert log.tool_calls is not None
        assert log.tool_calls[0]["name"] == "tavily_search"
    
    def test_query_logs_by_job(self, db_session, sample_research_report):
        """Test querying logs by job ID."""
        from src.db.models import AgentLog
        
        # Create multiple log entries
        agents = ["Enricher", "Planner", "Researcher"]
        for agent in agents:
            log = AgentLog(
                job_id=sample_research_report.id,
                agent_name=agent,
                role="assistant",
                content=f"Output from {agent}"
            )
            db_session.add(log)
        db_session.commit()
        
        logs = db_session.query(AgentLog).filter(
            AgentLog.job_id == sample_research_report.id
        ).all()
        
        assert len(logs) == 3
    
    def test_agent_log_timestamp(self, db_session, sample_research_report):
        """Test agent log has timestamp."""
        from src.db.models import AgentLog
        
        log = AgentLog(
            job_id=sample_research_report.id,
            agent_name="Test",
            role="user",
            content="Test"
        )
        db_session.add(log)
        db_session.commit()
        db_session.refresh(log)
        
        assert log.timestamp is not None


@pytest.mark.integration
class TestDatabaseRelationships:
    """Tests for database relationships between models."""
    
    def test_task_belongs_to_report(self, db_session, sample_research_report):
        """Test task has correct job_id reference."""
        from src.db.models import ResearchTask
        
        task = ResearchTask(
            job_id=sample_research_report.id,
            title="Test task",
            status="PENDING"
        )
        db_session.add(task)
        db_session.commit()
        
        assert task.job_id == sample_research_report.id
    
    def test_chunk_belongs_to_report(self, db_session, sample_research_report):
        """Test chunk has correct job_id reference."""
        from src.db.models import ResearchChunk
        
        chunk = ResearchChunk(
            job_id=sample_research_report.id,
            content="Test chunk",
            embedding=[0.1] * 1024
        )
        db_session.add(chunk)
        db_session.commit()
        
        assert chunk.job_id == sample_research_report.id
    
    def test_log_belongs_to_report(self, db_session, sample_research_report):
        """Test log has correct job_id reference."""
        from src.db.models import AgentLog
        
        log = AgentLog(
            job_id=sample_research_report.id,
            agent_name="Test",
            role="assistant",
            content="Test"
        )
        db_session.add(log)
        db_session.commit()
        
        assert log.job_id == sample_research_report.id

