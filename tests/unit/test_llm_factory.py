"""
Unit tests for src/llm/factory.py
Tests LLM provider factory logic.
"""
import pytest
from unittest.mock import patch, MagicMock

from src.llm.factory import get_llm_provider
from src.llm.base import LLMProvider
from src.llm.bedrock import BedrockProvider
from src.llm.openrouter import OpenRouterProvider


class TestLLMFactory:
    """Tests for the LLM provider factory."""
    
    @patch('src.llm.bedrock.boto3')
    def test_get_bedrock_provider(self, mock_boto3):
        """Test getting Bedrock provider."""
        mock_boto3.Session.return_value.client.return_value = MagicMock()
        mock_boto3.client.return_value = MagicMock()
        
        provider = get_llm_provider("bedrock")
        
        assert isinstance(provider, BedrockProvider)
        assert isinstance(provider, LLMProvider)
    
    @patch('src.llm.bedrock.boto3')
    def test_get_bedrock_provider_case_insensitive(self, mock_boto3):
        """Test Bedrock provider lookup is case insensitive."""
        mock_boto3.Session.return_value.client.return_value = MagicMock()
        mock_boto3.client.return_value = MagicMock()
        
        provider1 = get_llm_provider("Bedrock")
        provider2 = get_llm_provider("BEDROCK")
        
        assert isinstance(provider1, BedrockProvider)
        assert isinstance(provider2, BedrockProvider)
    
    @patch('src.llm.openrouter.OpenAI')
    def test_get_openrouter_provider(self, mock_openai):
        """Test getting OpenRouter provider."""
        mock_openai.return_value = MagicMock()
        
        provider = get_llm_provider("openrouter")
        
        assert isinstance(provider, OpenRouterProvider)
        assert isinstance(provider, LLMProvider)
    
    @patch('src.llm.openrouter.OpenAI')
    def test_get_openrouter_provider_case_insensitive(self, mock_openai):
        """Test OpenRouter provider lookup is case insensitive."""
        mock_openai.return_value = MagicMock()
        
        provider1 = get_llm_provider("OpenRouter")
        provider2 = get_llm_provider("OPENROUTER")
        
        assert isinstance(provider1, OpenRouterProvider)
        assert isinstance(provider2, OpenRouterProvider)
    
    def test_unknown_provider_raises_error(self):
        """Test unknown provider name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_llm_provider("unknown_provider")
        
        assert "Unknown provider" in str(exc_info.value)
        assert "unknown_provider" in str(exc_info.value)
    
    def test_empty_provider_name_raises_error(self):
        """Test empty provider name raises error."""
        with pytest.raises(ValueError):
            get_llm_provider("")
    
    @patch('src.llm.bedrock.boto3')
    def test_providers_implement_interface(self, mock_boto3):
        """Test all providers implement the LLMProvider interface."""
        mock_boto3.Session.return_value.client.return_value = MagicMock()
        mock_boto3.client.return_value = MagicMock()
        
        bedrock = get_llm_provider("bedrock")
        
        # Check interface methods exist
        assert hasattr(bedrock, 'generate')
        assert callable(bedrock.generate)
        assert hasattr(bedrock, 'get_embedding')
        assert callable(bedrock.get_embedding)

