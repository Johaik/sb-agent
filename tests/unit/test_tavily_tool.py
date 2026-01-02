"""
Unit tests for src/tools/tavily_tool.py
Tests Tavily search functionality with mocked client.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestTavilySearch:
    """Tests for the tavily_search function."""
    
    @pytest.fixture
    def mock_tavily_client(self):
        """Create a mock Tavily client."""
        with patch('src.tools.tavily_tool.TavilyClient') as mock:
            client_instance = MagicMock()
            mock.return_value = client_instance
            yield mock, client_instance
    
    def test_basic_search(self, mock_tavily_client):
        """Test basic search returns formatted results."""
        mock_class, mock_client = mock_tavily_client
        
        mock_client.search.return_value = {
            "answer": "Test answer",
            "results": [
                {"title": "Result 1", "url": "https://example.com/1", "content": "Content 1", "score": 0.9}
            ]
        }
        
        from src.tools.tavily_tool import tavily_search
        result = tavily_search(query="test query")
        
        assert "answer" in result
        assert "results" in result
        assert result["answer"] == "Test answer"
        assert len(result["results"]) == 1
    
    def test_search_with_parameters(self, mock_tavily_client):
        """Test search passes parameters correctly."""
        mock_class, mock_client = mock_tavily_client
        
        mock_client.search.return_value = {"answer": "", "results": []}
        
        from src.tools.tavily_tool import tavily_search
        tavily_search(
            query="test",
            search_depth="basic",
            max_results=10,
            include_raw_content=True
        )
        
        mock_client.search.assert_called_once_with(
            query="test",
            search_depth="basic",
            max_results=10,
            include_raw_content=True,
            include_answer=True
        )
    
    def test_search_error_handling(self, mock_tavily_client):
        """Test search returns error dict on exception."""
        mock_class, mock_client = mock_tavily_client
        
        mock_client.search.side_effect = Exception("API Error")
        
        from src.tools.tavily_tool import tavily_search
        result = tavily_search(query="test")
        
        assert "error" in result
        assert "API Error" in result["error"]
    
    def test_search_default_parameters(self, mock_tavily_client):
        """Test search uses correct default parameters."""
        mock_class, mock_client = mock_tavily_client
        
        mock_client.search.return_value = {"answer": "", "results": []}
        
        from src.tools.tavily_tool import tavily_search
        tavily_search(query="test query")
        
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["search_depth"] == "advanced"
        assert call_kwargs["max_results"] == 5
        assert call_kwargs["include_raw_content"] is False
        assert call_kwargs["include_answer"] is True
    
    def test_search_empty_results(self, mock_tavily_client):
        """Test search handles empty results."""
        mock_class, mock_client = mock_tavily_client
        
        mock_client.search.return_value = {"answer": "", "results": []}
        
        from src.tools.tavily_tool import tavily_search
        result = tavily_search(query="obscure query")
        
        assert result["answer"] == ""
        assert result["results"] == []


class TestDeepSearch:
    """Tests for the _run_deep_search function."""
    
    @pytest.fixture
    def mock_tavily_and_llm(self):
        """Mock both Tavily client and LLM provider."""
        with patch('src.tools.tavily_tool.TavilyClient') as mock_tavily, \
             patch('src.tools.tavily_tool.get_llm_provider') as mock_llm_factory:
            
            tavily_instance = MagicMock()
            mock_tavily.return_value = tavily_instance
            
            llm_instance = MagicMock()
            mock_llm_factory.return_value = llm_instance
            
            yield mock_tavily, tavily_instance, mock_llm_factory, llm_instance
    
    def test_deep_search_generates_subqueries(self, mock_tavily_and_llm):
        """Test deep search uses LLM to generate sub-queries."""
        mock_tavily, tavily_client, mock_llm_factory, llm = mock_tavily_and_llm
        
        # LLM generates sub-queries
        llm.generate.return_value = {
            "content": "What is quantum computing?\nHow does quantum entanglement work?\nQuantum computing applications"
        }
        
        # Tavily returns results for each query
        tavily_client.search.return_value = {
            "answer": "Quantum answer",
            "results": [{"title": "Result", "url": "https://example.com", "content": "Content", "score": 0.9}]
        }
        
        from src.tools.tavily_tool import tavily_search
        result = tavily_search(query="quantum computing", deep_search=True)
        
        # LLM should be called to generate sub-queries
        mock_llm_factory.assert_called_once_with("bedrock")
        llm.generate.assert_called_once()
        
        # Tavily should be called multiple times (original + sub-queries)
        assert tavily_client.search.call_count >= 1
    
    def test_deep_search_deduplicates_urls(self, mock_tavily_and_llm):
        """Test deep search removes duplicate URLs."""
        mock_tavily, tavily_client, mock_llm_factory, llm = mock_tavily_and_llm
        
        llm.generate.return_value = {"content": "Query 1\nQuery 2"}
        
        # Return same URL for different queries
        tavily_client.search.return_value = {
            "answer": "Answer",
            "results": [
                {"title": "Same Result", "url": "https://example.com/same", "content": "Content", "score": 0.9}
            ]
        }
        
        from src.tools.tavily_tool import tavily_search
        result = tavily_search(query="test", deep_search=True)
        
        # Should have only unique URLs
        urls = [r["url"] for r in result["results"]]
        assert len(urls) == len(set(urls))
    
    def test_deep_search_llm_fallback(self, mock_tavily_and_llm):
        """Test deep search falls back to original query on LLM error."""
        mock_tavily, tavily_client, mock_llm_factory, llm = mock_tavily_and_llm
        
        # LLM fails
        llm.generate.side_effect = Exception("LLM Error")
        
        tavily_client.search.return_value = {
            "answer": "Answer",
            "results": [{"title": "Result", "url": "https://example.com", "content": "Content", "score": 0.9}]
        }
        
        from src.tools.tavily_tool import tavily_search
        result = tavily_search(query="original query", deep_search=True)
        
        # Should still return results using original query
        assert "results" in result
        assert len(result["results"]) >= 1
    
    def test_deep_search_limits_queries(self, mock_tavily_and_llm):
        """Test deep search limits to 4 queries max."""
        mock_tavily, tavily_client, mock_llm_factory, llm = mock_tavily_and_llm
        
        # LLM returns many queries
        llm.generate.return_value = {
            "content": "Query 1\nQuery 2\nQuery 3\nQuery 4\nQuery 5\nQuery 6"
        }
        
        tavily_client.search.return_value = {
            "answer": "",
            "results": [{"title": "R", "url": "https://example.com/unique", "content": "C", "score": 0.9}]
        }
        
        from src.tools.tavily_tool import tavily_search
        result = tavily_search(query="test", deep_search=True)
        
        # Should be called at most 4 times (limit)
        assert tavily_client.search.call_count <= 4
    
    def test_deep_search_includes_original_query(self, mock_tavily_and_llm):
        """Test deep search always includes original query."""
        mock_tavily, tavily_client, mock_llm_factory, llm = mock_tavily_and_llm
        
        # LLM returns queries that don't include original
        llm.generate.return_value = {
            "content": "Different query 1\nDifferent query 2"
        }
        
        tavily_client.search.return_value = {
            "answer": "",
            "results": []
        }
        
        from src.tools.tavily_tool import tavily_search
        result = tavily_search(query="original query", deep_search=True)
        
        # First call should be with original query
        first_call_query = tavily_client.search.call_args_list[0].kwargs.get("query")
        assert first_call_query == "original query"
    
    def test_deep_search_search_error_continues(self, mock_tavily_and_llm):
        """Test deep search continues on individual search errors."""
        mock_tavily, tavily_client, mock_llm_factory, llm = mock_tavily_and_llm
        
        llm.generate.return_value = {"content": "Query 1\nQuery 2"}
        
        # First search fails, second succeeds
        tavily_client.search.side_effect = [
            Exception("First search failed"),
            {"answer": "Answer", "results": [{"title": "R", "url": "https://example.com", "content": "C", "score": 0.9}]}
        ]
        
        from src.tools.tavily_tool import tavily_search
        result = tavily_search(query="test", deep_search=True)
        
        # Should still return results from successful query
        assert "results" in result


class TestFormatOutput:
    """Tests for the _format_output function."""
    
    def test_format_output_basic(self):
        """Test basic output formatting."""
        from src.tools.tavily_tool import _format_output
        
        results = [
            {"title": "Test", "url": "https://example.com", "content": "Test content", "score": 0.9}
        ]
        
        output = _format_output("Test answer", results)
        
        assert output["answer"] == "Test answer"
        assert len(output["results"]) == 1
        assert output["results"][0]["title"] == "Test"
        assert output["results"][0]["url"] == "https://example.com"
        assert output["results"][0]["content"] == "Test content"
        assert output["results"][0]["score"] == 0.9
    
    def test_format_output_truncates_long_content(self):
        """Test content over 5000 chars is truncated."""
        from src.tools.tavily_tool import _format_output
        
        long_content = "x" * 6000
        results = [
            {"title": "Test", "url": "https://example.com", "content": long_content, "score": 0.9}
        ]
        
        output = _format_output("", results)
        
        assert len(output["results"][0]["content"]) < 6000
        assert output["results"][0]["content"].endswith("...(truncated)")
    
    def test_format_output_prefers_raw_content(self):
        """Test raw_content is used when available."""
        from src.tools.tavily_tool import _format_output
        
        results = [
            {
                "title": "Test",
                "url": "https://example.com",
                "content": "Short summary",
                "raw_content": "Full detailed content here",
                "score": 0.9
            }
        ]
        
        output = _format_output("", results)
        
        assert output["results"][0]["content"] == "Full detailed content here"
    
    def test_format_output_falls_back_to_content(self):
        """Test falls back to content when raw_content is None."""
        from src.tools.tavily_tool import _format_output
        
        results = [
            {
                "title": "Test",
                "url": "https://example.com",
                "content": "Regular content",
                "raw_content": None,
                "score": 0.9
            }
        ]
        
        output = _format_output("", results)
        
        assert output["results"][0]["content"] == "Regular content"
    
    def test_format_output_empty_results(self):
        """Test formatting empty results."""
        from src.tools.tavily_tool import _format_output
        
        output = _format_output("", [])
        
        assert output["answer"] == ""
        assert output["results"] == []
    
    def test_format_output_multiple_results(self):
        """Test formatting multiple results."""
        from src.tools.tavily_tool import _format_output
        
        results = [
            {"title": f"Result {i}", "url": f"https://example.com/{i}", "content": f"Content {i}", "score": 0.9 - i*0.1}
            for i in range(5)
        ]
        
        output = _format_output("Multi-result answer", results)
        
        assert len(output["results"]) == 5
        assert output["results"][0]["title"] == "Result 0"
        assert output["results"][4]["title"] == "Result 4"


class TestTavilyToolIntegration:
    """Integration-style tests for tavily_tool."""
    
    @pytest.fixture
    def mock_tavily_client(self):
        """Create a mock Tavily client."""
        with patch('src.tools.tavily_tool.TavilyClient') as mock:
            client_instance = MagicMock()
            mock.return_value = client_instance
            yield mock, client_instance
    
    def test_search_uses_config_api_key(self, mock_tavily_client):
        """Test search uses API key from Config."""
        mock_class, mock_client = mock_tavily_client
        mock_client.search.return_value = {"answer": "", "results": []}
        
        from src.tools.tavily_tool import tavily_search
        tavily_search(query="test")
        
        # Verify TavilyClient was instantiated (with api_key from Config)
        mock_class.assert_called_once()
    
    def test_search_result_structure(self, mock_tavily_client):
        """Test search result has consistent structure."""
        mock_class, mock_client = mock_tavily_client
        
        mock_client.search.return_value = {
            "answer": "The answer",
            "results": [
                {"title": "T1", "url": "https://a.com", "content": "C1", "score": 0.9},
                {"title": "T2", "url": "https://b.com", "content": "C2", "score": 0.8}
            ]
        }
        
        from src.tools.tavily_tool import tavily_search
        result = tavily_search(query="test")
        
        # Verify structure
        assert isinstance(result, dict)
        assert "answer" in result
        assert "results" in result
        assert isinstance(result["results"], list)
        
        for r in result["results"]:
            assert "title" in r
            assert "url" in r
            assert "content" in r
            assert "score" in r

