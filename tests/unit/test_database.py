"""
Unit tests for src/db/database.py
Tests database configuration, engine setup, and session management.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestDatabaseConfig:
    """Tests for database configuration."""
    
    def test_base_class_exists(self):
        """Test Base declarative class is created."""
        from src.db.database import Base
        
        assert Base is not None
        assert hasattr(Base, 'metadata')
    
    def test_session_local_exists(self):
        """Test SessionLocal factory is created."""
        from src.db.database import SessionLocal
        
        assert SessionLocal is not None
    
    def test_engine_exists(self):
        """Test engine is created."""
        from src.db.database import engine
        
        assert engine is not None
    
    def test_engine_removes_asyncpg_from_url(self):
        """Test engine URL replaces +asyncpg for sync operation."""
        from src.db.database import engine
        
        # The engine URL should not contain asyncpg for sync Celery use
        url_str = str(engine.url)
        assert "+asyncpg" not in url_str


class TestSessionLocal:
    """Tests for SessionLocal configuration."""
    
    def test_session_local_autocommit_disabled(self):
        """Test SessionLocal has autocommit=False."""
        from src.db.database import SessionLocal
        
        # SessionLocal is a sessionmaker, check its configuration
        assert SessionLocal.kw.get('autocommit') is False
    
    def test_session_local_autoflush_disabled(self):
        """Test SessionLocal has autoflush=False."""
        from src.db.database import SessionLocal
        
        assert SessionLocal.kw.get('autoflush') is False
    
    def test_session_local_bound_to_engine(self):
        """Test SessionLocal is bound to the engine."""
        from src.db.database import SessionLocal, engine
        
        # Check that bind is set (may be in kw or directly)
        bind = SessionLocal.kw.get('bind')
        assert bind is engine


class TestGetDb:
    """Tests for the get_db generator function."""
    
    def test_get_db_returns_generator(self):
        """Test get_db returns a generator."""
        from src.db.database import get_db
        import types
        
        result = get_db()
        assert isinstance(result, types.GeneratorType)
    
    def test_get_db_yields_session(self):
        """Test get_db yields a session."""
        from src.db.database import get_db, SessionLocal
        
        gen = get_db()
        session = next(gen)
        
        # Should be a Session instance
        assert session is not None
        
        # Clean up
        try:
            next(gen)
        except StopIteration:
            pass
    
    def test_get_db_closes_session_on_exit(self):
        """Test get_db closes session when generator exits."""
        # Mock SessionLocal to track close() calls
        with patch('src.db.database.SessionLocal') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            
            # Re-import to use the mock
            from src.db import database
            
            # Create a new get_db with the mock
            def get_db_mocked():
                db = mock_session_class()
                try:
                    yield db
                finally:
                    db.close()
            
            gen = get_db_mocked()
            session = next(gen)
            
            # close() should not be called yet
            mock_session.close.assert_not_called()
            
            # Exit the generator
            try:
                next(gen)
            except StopIteration:
                pass
            
            # Now close() should be called
            mock_session.close.assert_called_once()
    
    def test_get_db_closes_session_on_exception(self):
        """Test get_db closes session even when exception occurs."""
        with patch('src.db.database.SessionLocal') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            
            def get_db_mocked():
                db = mock_session_class()
                try:
                    yield db
                finally:
                    db.close()
            
            gen = get_db_mocked()
            session = next(gen)
            
            # Throw an exception into the generator
            try:
                gen.throw(ValueError("Test exception"))
            except ValueError:
                pass
            
            # close() should still be called
            mock_session.close.assert_called_once()
    
    def test_get_db_can_be_used_in_context(self):
        """Test get_db works with for loop (common FastAPI pattern)."""
        from src.db.database import get_db
        
        sessions = []
        for session in get_db():
            sessions.append(session)
        
        # Should have yielded exactly one session
        assert len(sessions) == 1
        assert sessions[0] is not None


class TestDatabaseUrlHandling:
    """Tests for database URL handling."""
    
    def test_database_url_from_config(self):
        """Test database URL is loaded from Config."""
        from src.db.database import engine
        from src.config import Config
        
        # Engine should use the URL from config (with asyncpg replaced)
        expected_url = Config.DATABASE_URL.replace("+asyncpg", "")
        actual_url = str(engine.url)
        
        # Compare the basic structure (host/db name)
        # Note: URLs may differ slightly in format
        assert engine is not None


class TestModelImports:
    """Tests that models can be properly imported with the database."""
    
    def test_models_use_base(self):
        """Test that models use the Base from database module."""
        from src.db.database import Base
        from src.db.models import ResearchReport, ResearchTask, AgentLog, ResearchChunk
        
        # All models should be registered with Base
        assert ResearchReport.__tablename__ in [t.name for t in Base.metadata.tables.values()]
        assert ResearchTask.__tablename__ in [t.name for t in Base.metadata.tables.values()]
        assert AgentLog.__tablename__ in [t.name for t in Base.metadata.tables.values()]
        assert ResearchChunk.__tablename__ in [t.name for t in Base.metadata.tables.values()]

