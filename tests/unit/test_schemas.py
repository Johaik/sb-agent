"""
Unit tests for src/api/schemas.py
Tests Pydantic schema validation and serialization.
"""
import pytest
from uuid import uuid4
from datetime import datetime
from pydantic import ValidationError

from src.api.schemas import (
    HealthResponse,
    ResearchRequest,
    ResearchJobStatus,
    ResearchResult
)


class TestHealthResponse:
    """Tests for HealthResponse schema."""
    
    def test_valid_health_response(self):
        """Test creating a valid health response."""
        response = HealthResponse(status="ok", details={"version": "1.0.0"})
        assert response.status == "ok"
        assert response.details["version"] == "1.0.0"
    
    def test_health_response_empty_details(self):
        """Test health response with empty details."""
        response = HealthResponse(status="ok")
        assert response.status == "ok"
        assert response.details == {}
    
    def test_health_response_degraded_status(self):
        """Test health response with degraded status."""
        response = HealthResponse(
            status="degraded",
            details={"database": "connection error"}
        )
        assert response.status == "degraded"
        assert "database" in response.details


class TestResearchRequest:
    """Tests for ResearchRequest schema."""
    
    def test_valid_research_request(self):
        """Test creating a valid research request."""
        request = ResearchRequest(idea="The impact of AI on healthcare")
        assert request.idea == "The impact of AI on healthcare"
    
    def test_research_request_minimum_length(self):
        """Test research request with minimum length idea."""
        request = ResearchRequest(idea="12345")  # Exactly 5 chars
        assert request.idea == "12345"
    
    def test_research_request_too_short(self):
        """Test research request with idea that's too short."""
        with pytest.raises(ValidationError) as exc_info:
            ResearchRequest(idea="1234")  # 4 chars, below minimum
        
        error = exc_info.value.errors()[0]
        assert error["type"] == "string_too_short"
    
    def test_research_request_empty_idea(self):
        """Test research request with empty idea."""
        with pytest.raises(ValidationError):
            ResearchRequest(idea="")
    
    def test_research_request_missing_idea(self):
        """Test research request without idea field."""
        with pytest.raises(ValidationError):
            ResearchRequest()
    
    def test_research_request_long_idea(self):
        """Test research request with a long idea."""
        long_idea = "A" * 1000
        request = ResearchRequest(idea=long_idea)
        assert len(request.idea) == 1000


class TestResearchJobStatus:
    """Tests for ResearchJobStatus schema."""
    
    def test_valid_job_status(self):
        """Test creating a valid job status."""
        job_id = uuid4()
        now = datetime.now()
        
        status = ResearchJobStatus(
            job_id=job_id,
            status="pending",
            progress_percent=0,
            current_phase="queued",
            created_at=now
        )
        
        assert status.job_id == job_id
        assert status.status == "pending"
        assert status.progress_percent == 0
        assert status.current_phase == "queued"
        assert status.created_at == now
    
    def test_job_status_processing(self):
        """Test job status in processing state."""
        status = ResearchJobStatus(
            job_id=uuid4(),
            status="processing",
            progress_percent=50,
            current_phase="researching",
            created_at=datetime.now()
        )
        
        assert status.status == "processing"
        assert status.progress_percent == 50
        assert status.current_phase == "researching"
    
    def test_job_status_completed(self):
        """Test job status in completed state."""
        now = datetime.now()
        status = ResearchJobStatus(
            job_id=uuid4(),
            status="completed",
            progress_percent=100,
            current_phase="reporting",
            created_at=now,
            updated_at=now
        )
        
        assert status.status == "completed"
        assert status.progress_percent == 100
    
    def test_job_status_failed_with_error(self):
        """Test job status in failed state with error message."""
        status = ResearchJobStatus(
            job_id=uuid4(),
            status="failed",
            progress_percent=0,
            current_phase="failed",
            created_at=datetime.now(),
            error="Connection timeout"
        )
        
        assert status.status == "failed"
        assert status.error == "Connection timeout"
    
    def test_job_status_progress_bounds_lower(self):
        """Test progress_percent minimum bound."""
        with pytest.raises(ValidationError) as exc_info:
            ResearchJobStatus(
                job_id=uuid4(),
                status="pending",
                progress_percent=-1,  # Below 0
                current_phase="queued",
                created_at=datetime.now()
            )
        
        error = exc_info.value.errors()[0]
        assert error["type"] == "greater_than_equal"
    
    def test_job_status_progress_bounds_upper(self):
        """Test progress_percent maximum bound."""
        with pytest.raises(ValidationError) as exc_info:
            ResearchJobStatus(
                job_id=uuid4(),
                status="pending",
                progress_percent=101,  # Above 100
                current_phase="queued",
                created_at=datetime.now()
            )
        
        error = exc_info.value.errors()[0]
        assert error["type"] == "less_than_equal"
    
    def test_job_status_optional_updated_at(self):
        """Test updated_at is optional."""
        status = ResearchJobStatus(
            job_id=uuid4(),
            status="pending",
            progress_percent=0,
            current_phase="queued",
            created_at=datetime.now()
        )
        
        assert status.updated_at is None


class TestResearchResult:
    """Tests for ResearchResult schema."""
    
    def test_valid_research_result(self):
        """Test creating a valid research result."""
        result = ResearchResult(
            job_id=uuid4(),
            status="completed",
            progress_percent=100,
            current_phase="reporting",
            created_at=datetime.now(),
            description="Detailed research description",
            report={
                "summary": "Research summary",
                "key_findings": ["Finding 1", "Finding 2"],
                "details": {"Section 1": "Details here"}
            }
        )
        
        assert result.status == "completed"
        assert result.description == "Detailed research description"
        assert result.report["summary"] == "Research summary"
    
    def test_research_result_inherits_job_status(self):
        """Test ResearchResult inherits from ResearchJobStatus."""
        result = ResearchResult(
            job_id=uuid4(),
            status="processing",
            progress_percent=50,
            current_phase="researching",
            created_at=datetime.now()
        )
        
        # Should have all parent fields
        assert hasattr(result, 'job_id')
        assert hasattr(result, 'status')
        assert hasattr(result, 'progress_percent')
        assert hasattr(result, 'current_phase')
        assert hasattr(result, 'created_at')
    
    def test_research_result_optional_fields(self):
        """Test optional description and report fields."""
        result = ResearchResult(
            job_id=uuid4(),
            status="pending",
            progress_percent=0,
            current_phase="queued",
            created_at=datetime.now()
        )
        
        assert result.description is None
        assert result.report is None
    
    def test_research_result_with_complex_report(self):
        """Test research result with complex nested report."""
        complex_report = {
            "summary": "A comprehensive summary with multiple paragraphs.",
            "key_findings": [
                "Finding 1 with details",
                "Finding 2 with more details",
                "Finding 3 with even more details"
            ],
            "details": {
                "Technical Analysis": {
                    "subsection1": "Content here",
                    "data": [1, 2, 3]
                },
                "Market Overview": "Market analysis content"
            },
            "metadata": {
                "sources": 15,
                "confidence": 0.85
            }
        }
        
        result = ResearchResult(
            job_id=uuid4(),
            status="completed",
            progress_percent=100,
            current_phase="reporting",
            created_at=datetime.now(),
            report=complex_report
        )
        
        assert result.report["summary"] == complex_report["summary"]
        assert len(result.report["key_findings"]) == 3
        assert "Technical Analysis" in result.report["details"]


class TestSchemasSerialization:
    """Tests for schema serialization/deserialization."""
    
    def test_research_request_json_schema(self):
        """Test ResearchRequest JSON schema generation."""
        schema = ResearchRequest.model_json_schema()
        
        assert "properties" in schema
        assert "idea" in schema["properties"]
    
    def test_job_status_serialization(self):
        """Test ResearchJobStatus serialization to dict."""
        job_id = uuid4()
        now = datetime.now()
        
        status = ResearchJobStatus(
            job_id=job_id,
            status="pending",
            progress_percent=0,
            current_phase="queued",
            created_at=now
        )
        
        data = status.model_dump()
        
        assert data["job_id"] == job_id
        assert data["status"] == "pending"
        assert data["progress_percent"] == 0
    
    def test_research_result_json_serialization(self):
        """Test ResearchResult JSON serialization."""
        result = ResearchResult(
            job_id=uuid4(),
            status="completed",
            progress_percent=100,
            current_phase="reporting",
            created_at=datetime.now(),
            report={"summary": "Test"}
        )
        
        json_str = result.model_dump_json()
        
        assert "completed" in json_str
        assert "summary" in json_str

