from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generates a response from the LLM.
        
        Args:
            messages: A list of message dictionaries (role, content).
            tools: A list of tool definitions.
            
        Returns:
            A dictionary containing the response content and any tool calls.
        """
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for the given text.
        
        Args:
            text: The text to embed.
            
        Returns:
            A list of floats representing the embedding.
        """
        pass
