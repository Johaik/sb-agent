from .bedrock import BedrockProvider
from .openrouter import OpenRouterProvider
from .base import LLMProvider

def get_llm_provider(provider_name: str) -> LLMProvider:
    if provider_name.lower() == "bedrock":
        return BedrockProvider()
    elif provider_name.lower() == "openrouter":
        return OpenRouterProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

