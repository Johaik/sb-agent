"""
Shared pytest fixtures for the Research Agent test suite.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set test environment variables BEFORE importing any src modules
os.environ["DATABASE_URL"] = "postgresql://test_user:test_password@localhost:5433/test_research_db"
os.environ["REDIS_URL"] = "redis://localhost:6380/0"
os.environ["TAVILY_API_KEY"] = "test-tavily-key"
os.environ["BEDROCK_REGION"] = "us-east-1"
os.environ["BEDROCK_PROFILE"] = "default"
os.environ["OPENROUTER_API_KEY"] = "test-openrouter-key"
os.environ["API_AUTH_ENABLED"] = "false"
os.environ["API_SECRET_KEY"] = "test-secret-key"


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "real_api: marks tests that require real API credentials (deselect with '-m \"not real_api\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks end-to-end tests"
    )


def pytest_collection_modifyitems(config, items):
    """Skip real_api tests by default unless explicitly requested."""
    if config.getoption("-m") and "real_api" in config.getoption("-m"):
        # real_api tests explicitly requested, don't skip
        return
    
    skip_real_api = pytest.mark.skip(reason="Real API tests skipped by default. Run with: pytest -m real_api")
    for item in items:
        if "real_api" in item.keywords:
            item.add_marker(skip_real_api)


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def db_engine():
    """Create a test database engine for the session."""
    from sqlalchemy import create_engine, text
    from src.db.database import Base
    
    engine = create_engine(os.environ["DATABASE_URL"])
    
    # Enable pgvector extension and create tables
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Cleanup: drop all tables after tests
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Create a new database session for each test with transaction rollback."""
    from sqlalchemy.orm import sessionmaker
    
    connection = db_engine.connect()
    transaction = connection.begin()
    
    Session = sessionmaker(bind=connection)
    session = Session()
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def override_get_db(db_session):
    """Override the get_db dependency for FastAPI testing."""
    def _get_db_override():
        yield db_session
    return _get_db_override


# ============================================================================
# FastAPI Test Client Fixtures
# ============================================================================

@pytest.fixture
def test_client(db_session, override_get_db, mock_redis):
    """Create a FastAPI test client with mocked dependencies."""
    from fastapi.testclient import TestClient
    from src.api.main import app, get_db
    
    app.dependency_overrides[get_db] = override_get_db
    
    with patch("src.api.main.redis_client", mock_redis):
        client = TestClient(app)
        yield client
    
    app.dependency_overrides.clear()


@pytest.fixture
def auth_test_client(db_session, override_get_db, mock_redis):
    """Create a FastAPI test client with authentication enabled."""
    from fastapi.testclient import TestClient
    from src.api.main import app, get_db
    
    # Enable auth for this test
    with patch.dict(os.environ, {"API_AUTH_ENABLED": "true"}):
        # Need to reimport config to pick up new value
        from src.config import Config
        original_auth = Config.API_AUTH_ENABLED
        Config.API_AUTH_ENABLED = True
        
        app.dependency_overrides[get_db] = override_get_db
        
        with patch("src.api.main.redis_client", mock_redis):
            client = TestClient(app)
            yield client
        
        app.dependency_overrides.clear()
        Config.API_AUTH_ENABLED = original_auth


# ============================================================================
# Mock Fixtures for External Services
# ============================================================================

@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis_mock = MagicMock()
    redis_mock.ping.return_value = True
    redis_mock.get.return_value = None
    redis_mock.setex.return_value = True
    return redis_mock


@pytest.fixture
def mock_bedrock_client():
    """Mock AWS Bedrock client."""
    import json
    
    client_mock = MagicMock()
    
    # Mock LLM response
    llm_response_body = MagicMock()
    llm_response_body.read.return_value = json.dumps({
        "content": [{"type": "text", "text": "This is a mocked response from Bedrock."}],
        "stop_reason": "end_turn"
    }).encode()
    
    client_mock.invoke_model.return_value = {"body": llm_response_body}
    
    return client_mock


@pytest.fixture
def mock_bedrock_embedding_client():
    """Mock AWS Bedrock client for embeddings."""
    import json
    
    client_mock = MagicMock()
    
    # Mock embedding response (1024 dimensions)
    embedding_response_body = MagicMock()
    embedding_response_body.read.return_value = json.dumps({
        "embedding": [0.1] * 1024
    }).encode()
    
    client_mock.invoke_model.return_value = {"body": embedding_response_body}
    
    return client_mock


@pytest.fixture
def mock_tavily_client():
    """Mock Tavily search client."""
    client_mock = MagicMock()
    
    client_mock.search.return_value = {
        "answer": "This is a mocked answer from Tavily.",
        "results": [
            {
                "title": "Test Result 1",
                "url": "https://example.com/1",
                "content": "This is test content from result 1.",
                "score": 0.95
            },
            {
                "title": "Test Result 2",
                "url": "https://example.com/2",
                "content": "This is test content from result 2.",
                "score": 0.90
            }
        ]
    }
    
    return client_mock


@pytest.fixture
def mock_openrouter_client():
    """Mock OpenRouter (OpenAI-compatible) client."""
    client_mock = MagicMock()
    
    # Create mock completion response
    mock_message = MagicMock()
    mock_message.content = "This is a mocked response from OpenRouter."
    mock_message.tool_calls = None
    
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    
    client_mock.chat.completions.create.return_value = mock_completion
    
    return client_mock


@pytest.fixture
def mock_strands_agent():
    """Mock Strands Agent for testing agent classes."""
    mock_result = MagicMock()
    mock_result.last_message = {
        "content": [{"text": "Mocked agent response"}]
    }
    
    agent_mock = MagicMock()
    agent_mock.return_value = mock_result
    
    return agent_mock


# ============================================================================
# Celery Fixtures
# ============================================================================

@pytest.fixture
def celery_app():
    """Get the Celery app for testing."""
    from src.worker.celery_app import celery_app
    
    # Configure for eager execution (synchronous)
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
    )
    
    return celery_app


@pytest.fixture
def celery_worker(celery_app):
    """Create a test Celery worker."""
    # For eager mode, we don't need an actual worker
    yield celery_app


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_research_idea():
    """Sample research idea for testing."""
    return "The impact of quantum computing on modern cryptography"


@pytest.fixture
def sample_job_id():
    """Sample job ID for testing."""
    return str(uuid4())


@pytest.fixture
def sample_research_report(db_session, sample_job_id):
    """Create a sample research report in the database."""
    from src.db.models import ResearchReport
    
    report = ResearchReport(
        id=sample_job_id,
        idea="Test research idea",
        description="Enriched test research description",
        status="pending"
    )
    db_session.add(report)
    db_session.commit()
    db_session.refresh(report)
    
    return report


@pytest.fixture
def sample_research_task(db_session, sample_research_report):
    """Create a sample research task in the database."""
    from src.db.models import ResearchTask
    
    task = ResearchTask(
        job_id=sample_research_report.id,
        title="Test research task",
        status="PENDING"
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)
    
    return task


@pytest.fixture
def completed_research_report(db_session):
    """Create a completed research report with full data."""
    from src.db.models import ResearchReport, ResearchTask
    
    job_id = uuid4()
    
    report = ResearchReport(
        id=job_id,
        idea="Completed test research",
        description="Full enriched description",
        status="completed",
        report={
            "summary": "This is a comprehensive summary.",
            "key_findings": ["Finding 1", "Finding 2"],
            "details": {
                "Section 1": "Detailed content for section 1.",
                "Section 2": "Detailed content for section 2."
            }
        }
    )
    db_session.add(report)
    
    # Add approved task
    task = ResearchTask(
        job_id=job_id,
        title="Completed task",
        status="APPROVED",
        result="Task research result"
    )
    db_session.add(task)
    
    db_session.commit()
    db_session.refresh(report)
    
    return report


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.generate.return_value = {
        "content": "Mocked LLM response",
        "tool_calls": None,
        "raw": {}
    }
    provider.get_embedding.return_value = [0.1] * 1024
    return provider


@pytest.fixture
def freeze_time():
    """Fixture to freeze time for testing."""
    from freezegun import freeze_time as ft
    return ft


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment after each test."""
    yield
    # Any cleanup needed after tests

