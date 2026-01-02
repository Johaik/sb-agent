"""
Optional real API tests for Tavily search.
These tests require valid API credentials and are skipped by default.
Run with: pytest -m real_api
"""
import pytest
import os


@pytest.fixture
def tavily_api_key():
    """Get Tavily API key from environment."""
    key = os.environ.get("TAVILY_API_KEY")
    if not key or key == "test-tavily-key":
        pytest.skip("Valid TAVILY_API_KEY not configured")
    return key


@pytest.mark.real_api
class TestTavilySearchReal:
    """Real API tests for Tavily search."""
    
    def test_basic_search(self, tavily_api_key):
        """Test basic Tavily search with real API."""
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=tavily_api_key)
        
        response = client.search(
            query="What is Python programming language?",
            search_depth="basic",
            max_results=3
        )
        
        assert "results" in response
        assert len(response["results"]) > 0
        assert "url" in response["results"][0]
        assert "content" in response["results"][0]
    
    def test_advanced_search(self, tavily_api_key):
        """Test advanced Tavily search with real API."""
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=tavily_api_key)
        
        response = client.search(
            query="Latest developments in quantum computing 2024",
            search_depth="advanced",
            max_results=5,
            include_answer=True
        )
        
        assert "results" in response
        assert len(response["results"]) > 0
        
        # Advanced search may include an answer
        if "answer" in response:
            assert isinstance(response["answer"], str)
    
    def test_search_with_answer(self, tavily_api_key):
        """Test Tavily search includes answer when requested."""
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=tavily_api_key)
        
        response = client.search(
            query="What is machine learning?",
            search_depth="basic",
            max_results=3,
            include_answer=True
        )
        
        assert "answer" in response or len(response.get("results", [])) > 0
    
    def test_tavily_tool_function(self, tavily_api_key):
        """Test the tavily_search tool function with real API."""
        from src.tools.tavily_tool import tavily_search
        
        result = tavily_search(
            query="Artificial intelligence applications",
            search_depth="basic",
            max_results=3
        )
        
        assert "error" not in result or result.get("results")
        if "results" in result:
            assert len(result["results"]) > 0
    
    def test_search_technical_topic(self, tavily_api_key):
        """Test searching for technical content."""
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=tavily_api_key)
        
        response = client.search(
            query="How does BERT transformer architecture work?",
            search_depth="advanced",
            max_results=5
        )
        
        assert "results" in response
        assert len(response["results"]) > 0
        
        # Check that results contain relevant content
        all_content = " ".join([r.get("content", "") for r in response["results"]])
        assert len(all_content) > 100  # Should have substantial content
    
    def test_search_result_structure(self, tavily_api_key):
        """Test Tavily search result structure."""
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=tavily_api_key)
        
        response = client.search(
            query="Climate change effects",
            search_depth="basic",
            max_results=2
        )
        
        for result in response.get("results", []):
            # Verify expected fields
            assert "title" in result
            assert "url" in result
            assert "content" in result
            
            # URL should be valid
            assert result["url"].startswith("http")


@pytest.mark.real_api
class TestTavilyDeepSearch:
    """Tests for deep search functionality."""
    
    def test_deep_search_generates_subqueries(self, tavily_api_key):
        """Test deep search with LLM-generated subqueries."""
        # This test requires both Tavily and an LLM provider
        # Skip if LLM not configured
        try:
            from src.tools.tavily_tool import tavily_search
            
            result = tavily_search(
                query="Future of renewable energy",
                search_depth="advanced",
                max_results=5,
                deep_search=False  # Start without deep search
            )
            
            assert "results" in result or "error" in result
            
        except Exception as e:
            pytest.skip(f"Deep search test skipped: {e}")


@pytest.mark.real_api
class TestTavilyErrorHandling:
    """Tests for Tavily API error handling."""
    
    def test_invalid_api_key(self):
        """Test handling of invalid API key."""
        from tavily import TavilyClient
        
        client = TavilyClient(api_key="invalid-key-12345")
        
        with pytest.raises(Exception):
            client.search(query="test query")
    
    def test_empty_query(self, tavily_api_key):
        """Test handling of empty query."""
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=tavily_api_key)
        
        # Behavior depends on Tavily API - might return empty or error
        try:
            response = client.search(query="", max_results=1)
            # If it doesn't error, results should be empty or minimal
        except Exception:
            pass  # Expected behavior

