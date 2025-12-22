import json
from openai import OpenAI
from typing import List, Dict, Any, Optional
from .base import LLMProvider
from ..config import Config

class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: str = Config.OPENROUTER_API_KEY, model: str = Config.OPENROUTER_MODEL):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        params = {
            "model": self.model,
            "messages": messages,
        }
        
        if tools:
            # OpenRouter/OpenAI tool format
            # Ensure tools are in the correct format { "type": "function", "function": ... }
            formatted_tools = []
            for t in tools:
                 # Check if it's already formatted or needs wrapping
                if "type" in t and t["type"] == "function":
                     formatted_tools.append(t)
                else:
                     formatted_tools.append({"type": "function", "function": t})
            
            params["tools"] = formatted_tools

        try:
            completion = self.client.chat.completions.create(**params)
            choice = completion.choices[0]
            message = choice.message
            
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    # Parse arguments if it's a string
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass # Keep as string if parsing fails
                            
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": args
                    }) 
            
            return {
                "content": message.content,
                "tool_calls": tool_calls if tool_calls else None,
                "raw": completion
            }

        except Exception as e:
            print(f"Error invoking OpenRouter: {e}")
            raise e

    def get_embedding(self, text: str) -> List[float]:
        # Simple OpenAI format embedding if supported, or raise error
        try:
            # Note: OpenRouter might route this to an appropriate model if configured
            # For now, we assume this provider is mainly for chat.
            # If user wanted embeddings via OpenAI compatible API:
            # response = self.client.embeddings.create(input=[text], model="text-embedding-3-small")
            # return response.data[0].embedding
            raise NotImplementedError("Embeddings not configured for OpenRouter provider yet.")
        except Exception as e:
            raise e
