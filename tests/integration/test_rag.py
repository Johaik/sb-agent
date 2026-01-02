"""
Integration tests for RAG (Retrieval-Augmented Generation) functionality.
Tests vector search with real pgvector.
"""
import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4


@pytest.mark.integration
class TestVectorSearch:
    """Tests for vector similarity search."""
    
    def test_search_returns_similar_chunks(self, db_session, sample_research_report):
        """Test search returns chunks ordered by similarity."""
        from src.db.models import ResearchChunk
        from src.db.vector import search_similar_chunks
        
        # Create chunks with different embeddings
        # Chunk 1: high similarity to query (similar embedding)
        chunk1 = ResearchChunk(
            job_id=sample_research_report.id,
            content="Quantum computing uses qubits for parallel processing",
            embedding=[0.9] * 1024  # High values
        )
        
        # Chunk 2: lower similarity
        chunk2 = ResearchChunk(
            job_id=sample_research_report.id,
            content="Traditional computers use binary bits",
            embedding=[0.1] * 1024  # Low values
        )
        
        # Chunk 3: medium similarity
        chunk3 = ResearchChunk(
            job_id=sample_research_report.id,
            content="Hybrid quantum-classical systems",
            embedding=[0.5] * 1024  # Medium values
        )
        
        db_session.add_all([chunk1, chunk2, chunk3])
        db_session.commit()
        
        # Search with high-value embedding (should match chunk1 best)
        query_embedding = [0.85] * 1024
        results = search_similar_chunks(db_session, query_embedding, limit=3)
        
        assert len(results) == 3
        # First result should be most similar
        assert "quantum" in results[0].content.lower() or "qubits" in results[0].content.lower()
    
    def test_search_respects_limit(self, db_session, sample_research_report):
        """Test search respects the limit parameter."""
        from src.db.models import ResearchChunk
        from src.db.vector import search_similar_chunks
        
        # Create 10 chunks
        for i in range(10):
            chunk = ResearchChunk(
                job_id=sample_research_report.id,
                content=f"Test chunk {i}",
                embedding=[0.1 * (i + 1)] * 1024
            )
            db_session.add(chunk)
        db_session.commit()
        
        query_embedding = [0.5] * 1024
        results = search_similar_chunks(db_session, query_embedding, limit=3)
        
        assert len(results) == 3
    
    def test_search_empty_database(self, db_session):
        """Test search returns empty when no chunks exist."""
        from src.db.vector import search_similar_chunks
        
        query_embedding = [0.5] * 1024
        results = search_similar_chunks(db_session, query_embedding, limit=5)
        
        assert results == []
    
    def test_search_single_chunk(self, db_session, sample_research_report):
        """Test search with only one chunk in database."""
        from src.db.models import ResearchChunk
        from src.db.vector import search_similar_chunks
        
        chunk = ResearchChunk(
            job_id=sample_research_report.id,
            content="Single chunk content",
            embedding=[0.5] * 1024
        )
        db_session.add(chunk)
        db_session.commit()
        
        query_embedding = [0.5] * 1024
        results = search_similar_chunks(db_session, query_embedding, limit=5)
        
        assert len(results) == 1
        assert results[0].content == "Single chunk content"


@pytest.mark.integration
class TestChunkStorage:
    """Tests for storing research chunks."""
    
    def test_save_chunks_from_dict_report(self, db_session, sample_research_report):
        """Test saving chunks from a dictionary report."""
        from src.db.models import ResearchChunk
        from src.db.vector import save_chunks
        
        mock_llm = MagicMock()
        mock_llm.get_embedding.return_value = [0.5] * 1024
        
        report = {
            "summary": "This is a comprehensive summary of the research findings that is long enough to be saved as a chunk.",
            "details": {
                "Technical Analysis": "This section contains detailed technical analysis with specific findings and data points."
            }
        }
        
        job_id = str(sample_research_report.id)
        save_chunks(db_session, job_id, report, mock_llm)
        
        chunks = db_session.query(ResearchChunk).filter(
            ResearchChunk.job_id == sample_research_report.id
        ).all()
        
        assert len(chunks) >= 1
        # Verify embedding was created
        for chunk in chunks:
            assert chunk.embedding is not None
    
    def test_save_chunks_from_string_report(self, db_session, sample_research_report):
        """Test saving chunks from a string report."""
        from src.db.models import ResearchChunk
        from src.db.vector import save_chunks
        
        mock_llm = MagicMock()
        mock_llm.get_embedding.return_value = [0.5] * 1024
        
        report = """This is the first paragraph of the report with enough content to be saved.

This is the second paragraph which should become another chunk because paragraphs are separated."""
        
        job_id = str(sample_research_report.id)
        save_chunks(db_session, job_id, report, mock_llm)
        
        chunks = db_session.query(ResearchChunk).filter(
            ResearchChunk.job_id == sample_research_report.id
        ).all()
        
        assert len(chunks) >= 1
    
    def test_save_chunks_skips_short_content(self, db_session, sample_research_report):
        """Test that very short chunks are skipped."""
        from src.db.models import ResearchChunk
        from src.db.vector import save_chunks
        
        mock_llm = MagicMock()
        mock_llm.get_embedding.return_value = [0.5] * 1024
        
        report = {
            "summary": "Short",  # Too short
            "details": {"S1": "Tiny"}  # Too short
        }
        
        job_id = str(sample_research_report.id)
        save_chunks(db_session, job_id, report, mock_llm)
        
        chunks = db_session.query(ResearchChunk).filter(
            ResearchChunk.job_id == sample_research_report.id
        ).all()
        
        assert len(chunks) == 0
    
    def test_save_chunks_handles_embedding_failure(self, db_session, sample_research_report):
        """Test save_chunks handles embedding failures gracefully."""
        from src.db.models import ResearchChunk
        from src.db.vector import save_chunks
        
        mock_llm = MagicMock()
        mock_llm.get_embedding.side_effect = Exception("Embedding API error")
        
        report = {
            "summary": "This is a long enough summary that would normally be saved as a chunk in the database."
        }
        
        job_id = str(sample_research_report.id)
        
        # Should not raise exception
        save_chunks(db_session, job_id, report, mock_llm)
        
        # No chunks should be saved due to embedding failure
        chunks = db_session.query(ResearchChunk).filter(
            ResearchChunk.job_id == sample_research_report.id
        ).all()
        
        assert len(chunks) == 0


@pytest.mark.integration
class TestRagSearchTool:
    """Tests for the rag_search tool function."""
    
    def test_rag_search_returns_results(self, db_session, sample_research_report):
        """Test rag_search tool returns formatted results."""
        from src.db.models import ResearchChunk
        
        # Create test chunks
        chunk = ResearchChunk(
            job_id=sample_research_report.id,
            content="AI is transforming healthcare through machine learning applications.",
            embedding=[0.5] * 1024
        )
        db_session.add(chunk)
        db_session.commit()
        
        with patch('src.tools.rag_tool.get_llm_provider') as mock_provider, \
             patch('src.tools.rag_tool.SessionLocal') as mock_session:
            
            mock_llm = MagicMock()
            mock_llm.get_embedding.return_value = [0.5] * 1024
            mock_provider.return_value = mock_llm
            
            mock_db = MagicMock()
            mock_result = MagicMock()
            mock_result.content = "AI is transforming healthcare"
            mock_db.query.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_result]
            mock_session.return_value = mock_db
            
            from src.tools.rag_tool import rag_search
            result = rag_search("healthcare AI")
            
            assert "[RAG]" in result
            assert "AI" in result or "healthcare" in result
    
    def test_rag_search_no_results(self):
        """Test rag_search when no relevant chunks found."""
        with patch('src.tools.rag_tool.get_llm_provider') as mock_provider, \
             patch('src.tools.rag_tool.SessionLocal') as mock_session:
            
            mock_llm = MagicMock()
            mock_llm.get_embedding.return_value = [0.5] * 1024
            mock_provider.return_value = mock_llm
            
            mock_db = MagicMock()
            mock_db.query.return_value.order_by.return_value.limit.return_value.all.return_value = []
            mock_session.return_value = mock_db
            
            from src.tools.rag_tool import rag_search
            result = rag_search("completely unrelated query xyz123")
            
            assert "No relevant information" in result
    
    def test_rag_search_embedding_error(self):
        """Test rag_search handles embedding generation errors."""
        with patch('src.tools.rag_tool.get_llm_provider') as mock_provider:
            mock_llm = MagicMock()
            mock_llm.get_embedding.side_effect = Exception("Embedding failed")
            mock_provider.return_value = mock_llm
            
            from src.tools.rag_tool import rag_search
            result = rag_search("test query")
            
            assert "Error" in result


@pytest.mark.integration
class TestRagIntegrationWithSearch:
    """Integration tests combining storage and retrieval."""
    
    def test_store_and_retrieve_cycle(self, db_session, sample_research_report):
        """Test complete store and retrieve cycle."""
        from src.db.models import ResearchChunk
        from src.db.vector import save_chunks, search_similar_chunks
        
        mock_llm = MagicMock()
        
        # Use consistent embeddings for testing
        call_count = [0]
        def get_embedding(text):
            call_count[0] += 1
            # Return slightly different embeddings based on content
            base = 0.5 if "quantum" in text.lower() else 0.3
            return [base] * 1024
        
        mock_llm.get_embedding.side_effect = get_embedding
        
        # Store research about quantum computing
        report = {
            "summary": "Quantum computing represents a revolutionary approach to computation using quantum mechanical phenomena.",
            "details": {
                "Applications": "Quantum computers excel at optimization problems and cryptography breaking tasks."
            }
        }
        
        job_id = str(sample_research_report.id)
        save_chunks(db_session, job_id, report, mock_llm)
        
        # Verify chunks were stored
        stored_chunks = db_session.query(ResearchChunk).filter(
            ResearchChunk.job_id == sample_research_report.id
        ).all()
        
        assert len(stored_chunks) >= 1
        
        # Search for related content
        query_embedding = [0.5] * 1024  # Similar to quantum content
        results = search_similar_chunks(db_session, query_embedding, limit=5)
        
        assert len(results) >= 1
        # Verify we can find the stored content
        all_content = " ".join([r.content for r in results])
        assert "quantum" in all_content.lower() or "computing" in all_content.lower()
    
    def test_cross_job_search(self, db_session):
        """Test that search finds chunks across different jobs."""
        from src.db.models import ResearchReport, ResearchChunk
        from src.db.vector import search_similar_chunks
        
        # Create two jobs with related content
        job1 = ResearchReport(idea="First research", status="completed")
        job2 = ResearchReport(idea="Second research", status="completed")
        db_session.add_all([job1, job2])
        db_session.commit()
        
        # Add chunks to both jobs
        chunk1 = ResearchChunk(
            job_id=job1.id,
            content="Machine learning algorithms for natural language processing",
            embedding=[0.6] * 1024
        )
        chunk2 = ResearchChunk(
            job_id=job2.id,
            content="Deep learning neural networks for image recognition",
            embedding=[0.65] * 1024
        )
        db_session.add_all([chunk1, chunk2])
        db_session.commit()
        
        # Search should find both
        query_embedding = [0.62] * 1024
        results = search_similar_chunks(db_session, query_embedding, limit=10)
        
        assert len(results) == 2
        
        # Both jobs should be represented
        job_ids = {r.job_id for r in results}
        assert job1.id in job_ids
        assert job2.id in job_ids


@pytest.mark.integration
class TestEmbeddingDimensions:
    """Tests for embedding dimension handling."""
    
    def test_chunk_with_1024_dim_embedding(self, db_session, sample_research_report):
        """Test storing chunk with correct 1024-dimension embedding."""
        from src.db.models import ResearchChunk
        
        embedding = [0.5] * 1024
        chunk = ResearchChunk(
            job_id=sample_research_report.id,
            content="Test content for embedding",
            embedding=embedding
        )
        db_session.add(chunk)
        db_session.commit()
        db_session.refresh(chunk)
        
        # Verify embedding was stored
        assert chunk.embedding is not None

