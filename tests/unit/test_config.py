"""
Unit tests for src/config.py
Tests environment variable loading and custom values.
"""
import os
import pytest
from unittest.mock import patch


class TestConfig:
    """Tests for the Config class."""
    
    def test_custom_bedrock_region(self):
        """Test custom Bedrock region is loaded from environment."""
        with patch.dict(os.environ, {"BEDROCK_REGION": "eu-west-1"}, clear=False):
            import importlib
            from src import config
            importlib.reload(config)
            
            assert config.Config.BEDROCK_REGION == "eu-west-1"
    
    def test_custom_bedrock_profile(self):
        """Test custom Bedrock profile is loaded from environment."""
        with patch.dict(os.environ, {"BEDROCK_PROFILE": "my-profile"}, clear=False):
            import importlib
            from src import config
            importlib.reload(config)
            
            assert config.Config.BEDROCK_PROFILE == "my-profile"
    
    def test_tavily_api_key_from_env(self):
        """Test Tavily API key is loaded from environment."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key-123"}, clear=False):
            import importlib
            from src import config
            importlib.reload(config)
            
            assert config.Config.TAVILY_API_KEY == "test-key-123"
    
    def test_database_url_custom(self):
        """Test custom database URL from environment."""
        custom_url = "postgresql://custom:pass@host:5432/mydb"
        with patch.dict(os.environ, {"DATABASE_URL": custom_url}, clear=False):
            import importlib
            from src import config
            importlib.reload(config)
            
            assert config.Config.DATABASE_URL == custom_url
    
    def test_redis_url_custom(self):
        """Test custom Redis URL from environment."""
        custom_url = "redis://custom-host:6379/1"
        with patch.dict(os.environ, {"REDIS_URL": custom_url}, clear=False):
            import importlib
            from src import config
            importlib.reload(config)
            
            assert config.Config.REDIS_URL == custom_url
    
    def test_api_auth_enabled(self):
        """Test API authentication can be enabled."""
        with patch.dict(os.environ, {"API_AUTH_ENABLED": "true"}, clear=False):
            import importlib
            from src import config
            importlib.reload(config)
            
            assert config.Config.API_AUTH_ENABLED is True
    
    def test_api_auth_enabled_case_insensitive(self):
        """Test API authentication enabled check is case insensitive."""
        with patch.dict(os.environ, {"API_AUTH_ENABLED": "TRUE"}, clear=False):
            import importlib
            from src import config
            importlib.reload(config)
            
            assert config.Config.API_AUTH_ENABLED is True
    
    def test_api_auth_disabled_explicit(self):
        """Test API authentication can be explicitly disabled."""
        with patch.dict(os.environ, {"API_AUTH_ENABLED": "false"}, clear=False):
            import importlib
            from src import config
            importlib.reload(config)
            
            assert config.Config.API_AUTH_ENABLED is False
    
    def test_api_secret_key_custom(self):
        """Test custom API secret key from environment."""
        with patch.dict(os.environ, {"API_SECRET_KEY": "my-secure-key"}, clear=False):
            import importlib
            from src import config
            importlib.reload(config)
            
            assert config.Config.API_SECRET_KEY == "my-secure-key"
    
    def test_openrouter_api_key(self):
        """Test OpenRouter API key from environment."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key-123"}, clear=False):
            import importlib
            from src import config
            importlib.reload(config)
            
            assert config.Config.OPENROUTER_API_KEY == "or-key-123"
    
    def test_openrouter_model_custom(self):
        """Test custom OpenRouter model from environment."""
        with patch.dict(os.environ, {"OPENROUTER_MODEL": "openai/gpt-4"}, clear=False):
            import importlib
            from src import config
            importlib.reload(config)
            
            assert config.Config.OPENROUTER_MODEL == "openai/gpt-4"
    
    def test_config_has_required_attributes(self):
        """Test Config class has all required attributes."""
        from src.config import Config
        
        assert hasattr(Config, 'BEDROCK_REGION')
        assert hasattr(Config, 'BEDROCK_PROFILE')
        assert hasattr(Config, 'OPENROUTER_API_KEY')
        assert hasattr(Config, 'OPENROUTER_MODEL')
        assert hasattr(Config, 'TAVILY_API_KEY')
        assert hasattr(Config, 'DATABASE_URL')
        assert hasattr(Config, 'REDIS_URL')
        assert hasattr(Config, 'API_AUTH_ENABLED')
        assert hasattr(Config, 'API_SECRET_KEY')

