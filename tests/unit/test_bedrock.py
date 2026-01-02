"""
Unit tests for src/llm/bedrock.py
Tests Bedrock provider with mocked AWS client.
"""
import pytest
import json
from unittest.mock import patch, MagicMock


class TestBedrockProvider:
    """Tests for BedrockProvider."""
    
    @pytest.fixture
    def mock_boto3(self):
        """Create a mock boto3 module."""
        with patch('src.llm.bedrock.boto3') as mock:
            mock_client = MagicMock()
            mock.Session.return_value.client.return_value = mock_client
            mock.client.return_value = mock_client
            yield mock, mock_client
    
    @pytest.fixture
    def bedrock_provider(self, mock_boto3):
        """Create a BedrockProvider with mocked client."""
        from src.llm.bedrock import BedrockProvider
        return BedrockProvider()
    
    def test_initialization(self, mock_boto3):
        """Test BedrockProvider initializes correctly."""
        mock, mock_client = mock_boto3
        from src.llm.bedrock import BedrockProvider
        
        provider = BedrockProvider()
        
        assert provider.model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert provider.embedding_model_id == "amazon.titan-embed-text-v2:0"
    
    def test_initialization_with_custom_region(self, mock_boto3):
        """Test BedrockProvider with custom region."""
        mock, mock_client = mock_boto3
        from src.llm.bedrock import BedrockProvider
        
        provider = BedrockProvider(region_name="eu-west-1")
        
        # Verify client was created with the region
        assert provider.client is not None
    
    def test_generate_simple_message(self, bedrock_provider, mock_boto3):
        """Test generating a simple text response."""
        mock, mock_client = mock_boto3
        
        # Mock response
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            "content": [{"type": "text", "text": "Hello, world!"}],
            "stop_reason": "end_turn"
        }).encode()
        mock_client.invoke_model.return_value = {"body": response_body}
        
        messages = [{"role": "user", "content": "Say hello"}]
        result = bedrock_provider.generate(messages)
        
        assert result["content"] == "Hello, world!"
        assert result["tool_calls"] is None
    
    def test_generate_with_system_prompt(self, bedrock_provider, mock_boto3):
        """Test generating with a system prompt."""
        mock, mock_client = mock_boto3
        
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            "content": [{"type": "text", "text": "Response"}],
            "stop_reason": "end_turn"
        }).encode()
        mock_client.invoke_model.return_value = {"body": response_body}
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
        
        result = bedrock_provider.generate(messages)
        
        # Verify invoke_model was called
        mock_client.invoke_model.assert_called_once()
        call_body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        
        assert call_body["system"] == "You are a helpful assistant."
    
    def test_generate_with_tool_calls(self, bedrock_provider, mock_boto3):
        """Test generating a response with tool calls."""
        mock, mock_client = mock_boto3
        
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            "content": [
                {"type": "text", "text": "I'll search for that."},
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "tavily_search",
                    "input": {"query": "AI news"}
                }
            ],
            "stop_reason": "tool_use"
        }).encode()
        mock_client.invoke_model.return_value = {"body": response_body}
        
        messages = [{"role": "user", "content": "Search for AI news"}]
        result = bedrock_provider.generate(messages)
        
        assert result["content"] == "I'll search for that."
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "tavily_search"
        assert result["tool_calls"][0]["id"] == "tool_123"
    
    def test_generate_with_tools_parameter(self, bedrock_provider, mock_boto3):
        """Test generating with tools parameter."""
        mock, mock_client = mock_boto3
        
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            "content": [{"type": "text", "text": "Response"}],
            "stop_reason": "end_turn"
        }).encode()
        mock_client.invoke_model.return_value = {"body": response_body}
        
        tools = [{
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}}
            }
        }]
        
        messages = [{"role": "user", "content": "Hello"}]
        result = bedrock_provider.generate(messages, tools=tools)
        
        call_body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        assert "tools" in call_body
        assert call_body["tools"][0]["name"] == "test_tool"
        assert "input_schema" in call_body["tools"][0]
    
    def test_generate_max_tokens(self, bedrock_provider, mock_boto3):
        """Test max_tokens parameter is passed correctly."""
        mock, mock_client = mock_boto3
        
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            "content": [{"type": "text", "text": "Response"}],
            "stop_reason": "end_turn"
        }).encode()
        mock_client.invoke_model.return_value = {"body": response_body}
        
        messages = [{"role": "user", "content": "Hello"}]
        result = bedrock_provider.generate(messages, max_tokens=8000)
        
        call_body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        assert call_body["max_tokens"] == 8000
    
    def test_get_embedding(self, bedrock_provider, mock_boto3):
        """Test getting embeddings."""
        mock, mock_client = mock_boto3
        
        expected_embedding = [0.1] * 1024
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            "embedding": expected_embedding
        }).encode()
        mock_client.invoke_model.return_value = {"body": response_body}
        
        result = bedrock_provider.get_embedding("Test text")
        
        assert result == expected_embedding
        assert len(result) == 1024
    
    def test_get_embedding_request_format(self, bedrock_provider, mock_boto3):
        """Test embedding request format."""
        mock, mock_client = mock_boto3
        
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            "embedding": [0.1] * 1024
        }).encode()
        mock_client.invoke_model.return_value = {"body": response_body}
        
        bedrock_provider.get_embedding("Test text")
        
        call_body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        assert call_body["inputText"] == "Test text"
        assert call_body["dimensions"] == 1024
        assert call_body["normalize"] is True
    
    def test_generate_error_handling(self, bedrock_provider, mock_boto3):
        """Test error handling in generate."""
        mock, mock_client = mock_boto3
        mock_client.invoke_model.side_effect = Exception("AWS error")
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(Exception) as exc_info:
            bedrock_provider.generate(messages)
        
        assert "AWS error" in str(exc_info.value)
    
    def test_get_embedding_error_handling(self, bedrock_provider, mock_boto3):
        """Test error handling in get_embedding."""
        mock, mock_client = mock_boto3
        mock_client.invoke_model.side_effect = Exception("Embedding error")
        
        with pytest.raises(Exception) as exc_info:
            bedrock_provider.get_embedding("Test text")
        
        assert "Embedding error" in str(exc_info.value)
    
    def test_convert_messages_with_tool_result(self, bedrock_provider, mock_boto3):
        """Test message conversion handles tool results."""
        mock, mock_client = mock_boto3
        
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            "content": [{"type": "text", "text": "Response"}],
            "stop_reason": "end_turn"
        }).encode()
        mock_client.invoke_model.return_value = {"body": response_body}
        
        messages = [
            {"role": "user", "content": "Search for something"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "123", "name": "search", "input": {"q": "test"}}]
            },
            {"role": "tool", "content": "Search results here", "tool_call_id": "123"},
        ]
        
        result = bedrock_provider.generate(messages)
        
        # Should not raise an error
        assert "content" in result


class TestBedrockProviderMessageConversion:
    """Tests for message conversion logic."""
    
    @pytest.fixture
    def bedrock_provider(self):
        """Create a BedrockProvider with mocked client."""
        with patch('src.llm.bedrock.boto3') as mock:
            mock_client = MagicMock()
            mock.Session.return_value.client.return_value = mock_client
            mock.client.return_value = mock_client
            
            from src.llm.bedrock import BedrockProvider
            return BedrockProvider()
    
    def test_convert_user_message(self, bedrock_provider):
        """Test converting simple user message."""
        messages = [{"role": "user", "content": "Hello"}]
        
        system, converted = bedrock_provider._convert_messages(messages)
        
        assert system == ""
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Hello"
    
    def test_convert_system_message(self, bedrock_provider):
        """Test converting system message."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ]
        
        system, converted = bedrock_provider._convert_messages(messages)
        
        assert system == "You are helpful."
        assert len(converted) == 1  # System message not in converted list
    
    def test_convert_multiple_system_messages(self, bedrock_provider):
        """Test multiple system messages are concatenated."""
        messages = [
            {"role": "system", "content": "Rule 1."},
            {"role": "system", "content": "Rule 2."},
            {"role": "user", "content": "Hello"}
        ]
        
        system, converted = bedrock_provider._convert_messages(messages)
        
        assert "Rule 1." in system
        assert "Rule 2." in system
    
    def test_convert_assistant_message_with_tool_calls(self, bedrock_provider):
        """Test converting assistant message with tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me search.",
                "tool_calls": [
                    {"id": "tc1", "name": "search", "input": {"query": "test"}}
                ]
            }
        ]
        
        system, converted = bedrock_provider._convert_messages(messages)
        
        assert len(converted) == 1
        content = converted[0]["content"]
        
        # Should have both text and tool_use blocks
        assert any(b.get("type") == "text" for b in content)
        assert any(b.get("type") == "tool_use" for b in content)

