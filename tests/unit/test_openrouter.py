"""
Unit tests for src/llm/openrouter.py
Tests OpenRouter provider with mocked OpenAI client.
"""
import pytest
import json
from unittest.mock import patch, MagicMock


class TestOpenRouterProvider:
    """Tests for OpenRouterProvider."""
    
    @pytest.fixture
    def mock_openai(self):
        """Create a mock OpenAI client."""
        with patch('src.llm.openrouter.OpenAI') as mock:
            mock_client = MagicMock()
            mock.return_value = mock_client
            yield mock, mock_client
    
    @pytest.fixture
    def openrouter_provider(self, mock_openai):
        """Create an OpenRouterProvider with mocked client."""
        from src.llm.openrouter import OpenRouterProvider
        return OpenRouterProvider()
    
    def test_initialization(self, mock_openai):
        """Test OpenRouterProvider initializes correctly."""
        mock, mock_client = mock_openai
        from src.llm.openrouter import OpenRouterProvider
        
        provider = OpenRouterProvider()
        
        # Verify OpenAI client was created with correct base URL
        mock.assert_called_once()
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
    
    def test_initialization_with_custom_model(self, mock_openai):
        """Test OpenRouterProvider with custom model."""
        mock, mock_client = mock_openai
        from src.llm.openrouter import OpenRouterProvider
        
        provider = OpenRouterProvider(model="openai/gpt-4")
        
        assert provider.model == "openai/gpt-4"
    
    def test_generate_simple_message(self, openrouter_provider, mock_openai):
        """Test generating a simple text response."""
        mock, mock_client = mock_openai
        
        # Mock response
        mock_message = MagicMock()
        mock_message.content = "Hello, world!"
        mock_message.tool_calls = None
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        messages = [{"role": "user", "content": "Say hello"}]
        result = openrouter_provider.generate(messages)
        
        assert result["content"] == "Hello, world!"
        assert result["tool_calls"] is None
    
    def test_generate_with_tools(self, openrouter_provider, mock_openai):
        """Test generating with tools parameter."""
        mock, mock_client = mock_openai
        
        mock_message = MagicMock()
        mock_message.content = "I'll search for that."
        mock_message.tool_calls = None
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        tools = [{
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}}
        }]
        
        messages = [{"role": "user", "content": "Search for AI"}]
        result = openrouter_provider.generate(messages, tools=tools)
        
        # Verify tools were passed correctly
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "tools" in call_kwargs
    
    def test_generate_with_tool_calls_response(self, openrouter_provider, mock_openai):
        """Test generating response with tool calls."""
        mock, mock_client = mock_openai
        
        # Mock tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "search"
        mock_tool_call.function.arguments = '{"query": "AI news"}'
        
        mock_message = MagicMock()
        mock_message.content = "Searching..."
        mock_message.tool_calls = [mock_tool_call]
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        messages = [{"role": "user", "content": "Search for AI"}]
        result = openrouter_provider.generate(messages)
        
        assert result["content"] == "Searching..."
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "search"
        assert result["tool_calls"][0]["id"] == "call_123"
        assert result["tool_calls"][0]["input"] == {"query": "AI news"}
    
    def test_generate_tool_call_with_invalid_json(self, openrouter_provider, mock_openai):
        """Test handling of invalid JSON in tool call arguments."""
        mock, mock_client = mock_openai
        
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "search"
        mock_tool_call.function.arguments = "invalid json{"  # Invalid JSON
        
        mock_message = MagicMock()
        mock_message.content = ""
        mock_message.tool_calls = [mock_tool_call]
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        messages = [{"role": "user", "content": "Test"}]
        result = openrouter_provider.generate(messages)
        
        # Should not raise, keeps as string
        assert result["tool_calls"][0]["input"] == "invalid json{"
    
    def test_generate_formats_tools_correctly(self, openrouter_provider, mock_openai):
        """Test tools are formatted correctly for OpenAI API."""
        mock, mock_client = mock_openai
        
        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.tool_calls = None
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Tool without "type" wrapper
        tools = [{
            "name": "search",
            "description": "Search",
            "parameters": {}
        }]
        
        openrouter_provider.generate([{"role": "user", "content": "Test"}], tools=tools)
        
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        formatted_tools = call_kwargs["tools"]
        
        # Should be wrapped in {"type": "function", "function": ...}
        assert formatted_tools[0]["type"] == "function"
        assert formatted_tools[0]["function"]["name"] == "search"
    
    def test_generate_preserves_already_formatted_tools(self, openrouter_provider, mock_openai):
        """Test already-formatted tools are preserved."""
        mock, mock_client = mock_openai
        
        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.tool_calls = None
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Already formatted tool
        tools = [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search",
                "parameters": {}
            }
        }]
        
        openrouter_provider.generate([{"role": "user", "content": "Test"}], tools=tools)
        
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        formatted_tools = call_kwargs["tools"]
        
        # Should remain as-is
        assert formatted_tools[0]["type"] == "function"
        assert "function" in formatted_tools[0]
    
    def test_generate_error_handling(self, openrouter_provider, mock_openai):
        """Test error handling in generate."""
        mock, mock_client = mock_openai
        mock_client.chat.completions.create.side_effect = Exception("API error")
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(Exception) as exc_info:
            openrouter_provider.generate(messages)
        
        assert "API error" in str(exc_info.value)
    
    def test_get_embedding_not_implemented(self, openrouter_provider):
        """Test get_embedding raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            openrouter_provider.get_embedding("Test text")
    
    def test_generate_passes_model(self, openrouter_provider, mock_openai):
        """Test model is passed to the API."""
        mock, mock_client = mock_openai
        
        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.tool_calls = None
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        openrouter_provider.generate([{"role": "user", "content": "Test"}])
        
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "model" in call_kwargs


class TestOpenRouterProviderIntegration:
    """Integration-style tests for OpenRouterProvider."""
    
    @pytest.fixture
    def provider_with_mock(self):
        """Create provider with comprehensive mock."""
        with patch('src.llm.openrouter.OpenAI') as mock:
            mock_client = MagicMock()
            mock.return_value = mock_client
            
            from src.llm.openrouter import OpenRouterProvider
            provider = OpenRouterProvider(api_key="test-key", model="test-model")
            
            yield provider, mock_client
    
    def test_multi_turn_conversation(self, provider_with_mock):
        """Test handling multi-turn conversation."""
        provider, mock_client = provider_with_mock
        
        mock_message = MagicMock()
        mock_message.content = "I understand. Here's more info."
        mock_message.tool_calls = None
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Tell me more."}
        ]
        
        result = provider.generate(messages)
        
        # Verify all messages were passed
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert len(call_kwargs["messages"]) == 3

