"""
Unit tests for src/db/vector.py
Tests vector operations and chunk management.
"""
import pytest
from unittest.mock import MagicMock, patch


class TestSearchSimilarChunks:
    """Tests for search_similar_chunks function."""
    
    def test_search_returns_results(self):
        """Test search returns chunks ordered by similarity."""
        from src.db.vector import search_similar_chunks
        
        # Create mock chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.content = "Quantum computing basics"
        
        mock_chunk2 = MagicMock()
        mock_chunk2.content = "Advanced quantum algorithms"
        
        # Mock the database query
        mock_db = MagicMock()
        mock_query = MagicMock()
        mock_query.order_by.return_value.limit.return_value.all.return_value = [
            mock_chunk1, mock_chunk2
        ]
        mock_db.query.return_value = mock_query
        
        query_embedding = [0.1] * 1024
        results = search_similar_chunks(mock_db, query_embedding, limit=5)
        
        assert len(results) == 2
        assert results[0].content == "Quantum computing basics"
    
    def test_search_with_custom_limit(self):
        """Test search respects limit parameter."""
        from src.db.vector import search_similar_chunks
        
        mock_db = MagicMock()
        mock_query = MagicMock()
        mock_query.order_by.return_value.limit.return_value.all.return_value = []
        mock_db.query.return_value = mock_query
        
        query_embedding = [0.1] * 1024
        search_similar_chunks(mock_db, query_embedding, limit=10)
        
        # Verify limit was called with correct value
        mock_query.order_by.return_value.limit.assert_called_with(10)
    
    def test_search_returns_empty_when_no_matches(self):
        """Test search returns empty list when no chunks match."""
        from src.db.vector import search_similar_chunks
        
        mock_db = MagicMock()
        mock_query = MagicMock()
        mock_query.order_by.return_value.limit.return_value.all.return_value = []
        mock_db.query.return_value = mock_query
        
        query_embedding = [0.1] * 1024
        results = search_similar_chunks(mock_db, query_embedding)
        
        assert results == []


class TestSaveChunks:
    """Tests for save_chunks function."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider for embeddings."""
        provider = MagicMock()
        provider.get_embedding.return_value = [0.1] * 1024
        return provider
    
    def test_save_chunks_with_dict_report(self, mock_llm_provider):
        """Test saving chunks from a dict report."""
        from src.db.vector import save_chunks
        
        mock_db = MagicMock()
        
        report_data = {
            "summary": "This is a test summary that is long enough to be saved as a chunk.",
            "details": {
                "Section 1": "This is the content for section 1 which should be saved as a separate chunk."
            }
        }
        
        save_chunks(mock_db, "job-123", report_data, mock_llm_provider)
        
        # Verify chunks were added
        mock_db.add_all.assert_called_once()
        mock_db.commit.assert_called_once()
        
        # Check chunks were created
        chunks = mock_db.add_all.call_args[0][0]
        assert len(chunks) >= 1  # At least one chunk from summary and section
    
    def test_save_chunks_with_string_report(self, mock_llm_provider):
        """Test saving chunks from a string report."""
        from src.db.vector import save_chunks
        
        mock_db = MagicMock()
        
        report_data = """This is a long text report.

This is the second paragraph that should become a separate chunk because it's separated by double newlines.

And this is the third paragraph with more content to save."""
        
        save_chunks(mock_db, "job-123", report_data, mock_llm_provider)
        
        mock_db.add_all.assert_called_once()
        chunks = mock_db.add_all.call_args[0][0]
        assert len(chunks) >= 1
    
    def test_save_chunks_skips_short_chunks(self, mock_llm_provider):
        """Test that very short chunks are skipped."""
        from src.db.vector import save_chunks
        
        mock_db = MagicMock()
        
        # All chunks too short (< 50 chars)
        report_data = {
            "summary": "Short",
            "details": {"S1": "Tiny"}
        }
        
        save_chunks(mock_db, "job-123", report_data, mock_llm_provider)
        
        # With no valid chunks, add_all might not be called or called with empty list
        if mock_db.add_all.called:
            chunks = mock_db.add_all.call_args[0][0]
            assert len(chunks) == 0
    
    def test_save_chunks_handles_list_details(self, mock_llm_provider):
        """Test handling details as a list instead of dict."""
        from src.db.vector import save_chunks
        
        mock_db = MagicMock()
        
        report_data = {
            "summary": "This is a summary that is definitely long enough to be saved as a chunk in the database.",
            "details": [
                "First detail item that is long enough to be saved as a chunk.",
                "Second detail item that is also long enough to be saved."
            ]
        }
        
        save_chunks(mock_db, "job-123", report_data, mock_llm_provider)
        
        mock_db.add_all.assert_called_once()
    
    def test_save_chunks_handles_embedding_error(self, mock_llm_provider):
        """Test handling embedding generation errors."""
        from src.db.vector import save_chunks
        
        mock_db = MagicMock()
        mock_llm_provider.get_embedding.side_effect = Exception("Embedding failed")
        
        report_data = {
            "summary": "This is a long enough summary to be saved as a chunk in the database for testing."
        }
        
        # Should not raise, just skip failed chunks
        save_chunks(mock_db, "job-123", report_data, mock_llm_provider)
        
        # No chunks should be added due to errors
        # add_all might be called with empty list or not at all
        if mock_db.add_all.called:
            chunks = mock_db.add_all.call_args[0][0]
            assert len(chunks) == 0
    
    def test_save_chunks_creates_correct_model(self, mock_llm_provider):
        """Test that ResearchChunk models are created correctly."""
        from src.db.vector import save_chunks
        from src.db.models import ResearchChunk
        
        mock_db = MagicMock()
        
        report_data = {
            "summary": "This is a comprehensive summary that should definitely be saved as a chunk in the database."
        }
        
        save_chunks(mock_db, "job-123", report_data, mock_llm_provider)
        
        mock_db.add_all.assert_called_once()
        chunks = mock_db.add_all.call_args[0][0]
        
        for chunk in chunks:
            assert isinstance(chunk, ResearchChunk)
            assert chunk.job_id == "job-123"
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 1024
    
    def test_save_chunks_with_key_findings(self, mock_llm_provider):
        """Test saving chunks includes key_findings if present."""
        from src.db.vector import save_chunks
        
        mock_db = MagicMock()
        
        report_data = {
            "summary": "This is a long summary for the research report that needs to be saved.",
            "key_findings": [
                "Finding 1 with enough content to be saved as a chunk in the database.",
                "Finding 2 with additional details that should also be saved."
            ],
            "details": {
                "Analysis": "Detailed analysis content that is long enough to be saved as a chunk."
            }
        }
        
        save_chunks(mock_db, "job-123", report_data, mock_llm_provider)
        
        mock_db.add_all.assert_called_once()


class TestVectorOperationsIntegration:
    """Integration-style tests for vector operations."""
    
    def test_chunk_content_format(self):
        """Test that chunk content is properly formatted."""
        from src.db.vector import save_chunks
        
        mock_db = MagicMock()
        mock_provider = MagicMock()
        mock_provider.get_embedding.return_value = [0.1] * 1024
        
        report_data = {
            "summary": "Quantum computing represents a paradigm shift in computational capability.",
            "details": {
                "Technical Overview": "Quantum computers use qubits which can exist in superposition states, enabling parallel computation."
            }
        }
        
        save_chunks(mock_db, "job-123", report_data, mock_provider)
        
        if mock_db.add_all.called:
            chunks = mock_db.add_all.call_args[0][0]
            for chunk in chunks:
                # Content should be readable text
                assert isinstance(chunk.content, str)
                assert len(chunk.content) > 0
    
    def test_embedding_dimensions(self):
        """Test that embeddings have correct dimensions."""
        from src.db.vector import save_chunks
        
        mock_db = MagicMock()
        mock_provider = MagicMock()
        mock_provider.get_embedding.return_value = [0.5] * 1024
        
        report_data = {
            "summary": "A long summary that will definitely be saved because it exceeds the minimum character threshold."
        }
        
        save_chunks(mock_db, "job-123", report_data, mock_provider)
        
        if mock_db.add_all.called:
            chunks = mock_db.add_all.call_args[0][0]
            for chunk in chunks:
                assert len(chunk.embedding) == 1024

