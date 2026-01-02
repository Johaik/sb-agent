"""
Optional real API tests for AWS Bedrock.
These tests require valid AWS credentials and are skipped by default.
Run with: pytest -m real_api
"""
import pytest
import os


@pytest.fixture
def bedrock_configured():
    """Check if Bedrock is properly configured."""
    # Check for AWS credentials
    has_creds = (
        os.environ.get("AWS_ACCESS_KEY_ID") or 
        os.environ.get("AWS_PROFILE") or
        os.environ.get("BEDROCK_PROFILE")
    )
    
    if not has_creds:
        pytest.skip("AWS credentials not configured for Bedrock")
    
    return True


@pytest.fixture
def bedrock_provider(bedrock_configured):
    """Create a real Bedrock provider."""
    from src.llm.bedrock import BedrockProvider
    return BedrockProvider()


@pytest.mark.real_api
class TestBedrockGenerateReal:
    """Real API tests for Bedrock text generation."""
    
    def test_simple_generation(self, bedrock_provider):
        """Test simple text generation with Bedrock."""
        messages = [
            {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
        ]
        
        result = bedrock_provider.generate(messages, max_tokens=50)
        
        assert "content" in result
        assert "4" in result["content"]
    
    def test_generation_with_system_prompt(self, bedrock_provider):
        """Test generation with system prompt."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is Python?"}
        ]
        
        result = bedrock_provider.generate(messages, max_tokens=200)
        
        assert "content" in result
        assert len(result["content"]) > 0
        assert "programming" in result["content"].lower() or "language" in result["content"].lower()
    
    def test_multi_turn_conversation(self, bedrock_provider):
        """Test multi-turn conversation."""
        messages = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
            {"role": "user", "content": "What is my name?"}
        ]
        
        result = bedrock_provider.generate(messages, max_tokens=100)
        
        assert "content" in result
        assert "Alice" in result["content"]
    
    def test_json_output(self, bedrock_provider):
        """Test requesting JSON output."""
        messages = [
            {"role": "system", "content": "You must respond with valid JSON only."},
            {"role": "user", "content": 'Create a JSON object with keys "name" and "age" for a person named John who is 30.'}
        ]
        
        result = bedrock_provider.generate(messages, max_tokens=100)
        
        assert "content" in result
        content = result["content"]
        
        # Should contain JSON-like content
        assert "{" in content
        assert "name" in content.lower() or "John" in content
    
    def test_long_output(self, bedrock_provider):
        """Test generating longer output."""
        messages = [
            {"role": "user", "content": "Write a brief paragraph about the importance of software testing."}
        ]
        
        result = bedrock_provider.generate(messages, max_tokens=500)
        
        assert "content" in result
        assert len(result["content"]) > 100
        assert "test" in result["content"].lower()


@pytest.mark.real_api
class TestBedrockEmbeddingsReal:
    """Real API tests for Bedrock embeddings."""
    
    def test_get_embedding(self, bedrock_provider):
        """Test getting embeddings from Bedrock."""
        text = "This is a test sentence for embedding generation."
        
        embedding = bedrock_provider.get_embedding(text)
        
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) == 1024  # Titan embed v2 dimension
        assert all(isinstance(x, (int, float)) for x in embedding)
    
    def test_embedding_similarity(self, bedrock_provider):
        """Test that similar texts have similar embeddings."""
        text1 = "Machine learning is a subset of artificial intelligence."
        text2 = "AI includes machine learning as one of its branches."
        text3 = "The weather today is sunny and warm."
        
        emb1 = bedrock_provider.get_embedding(text1)
        emb2 = bedrock_provider.get_embedding(text2)
        emb3 = bedrock_provider.get_embedding(text3)
        
        # Calculate cosine similarity
        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x ** 2 for x in a) ** 0.5
            norm_b = sum(x ** 2 for x in b) ** 0.5
            return dot / (norm_a * norm_b)
        
        sim_related = cosine_sim(emb1, emb2)
        sim_unrelated = cosine_sim(emb1, emb3)
        
        # Similar texts should have higher similarity
        assert sim_related > sim_unrelated
    
    def test_embedding_determinism(self, bedrock_provider):
        """Test that same text produces same embedding."""
        text = "Consistent embedding test."
        
        emb1 = bedrock_provider.get_embedding(text)
        emb2 = bedrock_provider.get_embedding(text)
        
        # Should be identical (or very close)
        for v1, v2 in zip(emb1, emb2):
            assert abs(v1 - v2) < 0.001
    
    def test_embedding_different_texts(self, bedrock_provider):
        """Test that different texts produce different embeddings."""
        text1 = "First unique text for testing."
        text2 = "Completely different second text."
        
        emb1 = bedrock_provider.get_embedding(text1)
        emb2 = bedrock_provider.get_embedding(text2)
        
        # Should not be identical
        differences = sum(1 for v1, v2 in zip(emb1, emb2) if abs(v1 - v2) > 0.01)
        assert differences > 100  # Most dimensions should differ


@pytest.mark.real_api
class TestBedrockToolUse:
    """Tests for Bedrock tool/function calling."""
    
    def test_tool_use_response(self, bedrock_provider):
        """Test that Bedrock can request tool use."""
        tools = [{
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["location"]
            }
        }]
        
        messages = [
            {"role": "user", "content": "What's the weather like in Paris?"}
        ]
        
        result = bedrock_provider.generate(messages, tools=tools, max_tokens=200)
        
        # Should either have tool_calls or content
        assert result.get("tool_calls") or result.get("content")


@pytest.mark.real_api
class TestBedrockErrorHandling:
    """Tests for Bedrock error handling."""
    
    def test_invalid_model_id(self, bedrock_configured):
        """Test handling of invalid model ID."""
        from src.llm.bedrock import BedrockProvider
        
        provider = BedrockProvider()
        provider.model_id = "invalid.model.id"
        
        messages = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(Exception):
            provider.generate(messages)
    
    def test_empty_messages(self, bedrock_provider):
        """Test handling of empty messages."""
        with pytest.raises(Exception):
            bedrock_provider.generate([])


@pytest.mark.real_api
class TestLLMProviderFactory:
    """Tests for LLM provider factory with real APIs."""
    
    def test_get_bedrock_provider(self, bedrock_configured):
        """Test factory creates working Bedrock provider."""
        from src.llm.factory import get_llm_provider
        
        provider = get_llm_provider("bedrock")
        
        # Test with simple generation
        messages = [{"role": "user", "content": "Say hello"}]
        result = provider.generate(messages, max_tokens=50)
        
        assert "content" in result
        assert len(result["content"]) > 0


@pytest.mark.real_api
class TestOpenRouterReal:
    """Real API tests for OpenRouter (optional)."""
    
    @pytest.fixture
    def openrouter_configured(self):
        """Check if OpenRouter is properly configured."""
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key or key == "test-openrouter-key":
            pytest.skip("OPENROUTER_API_KEY not configured")
        return True
    
    def test_openrouter_generation(self, openrouter_configured):
        """Test OpenRouter text generation."""
        from src.llm.openrouter import OpenRouterProvider
        
        provider = OpenRouterProvider()
        
        messages = [
            {"role": "user", "content": "What is 1 + 1? Answer with just the number."}
        ]
        
        result = provider.generate(messages)
        
        assert "content" in result
        assert "2" in result["content"]

