from abc import ABC, abstractmethod
from typing import Dict, Any, Callable

class Tool(ABC):
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

    @abstractmethod
    def run(self, invocation_state: Dict[str, Any] = None, **kwargs) -> Any:
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Returns the tool definition in a generic format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                 "type": "object",
                 "properties": self.parameters.get("properties", {}),
                 "required": self.parameters.get("required", [])
            }
        }

class FunctionTool(Tool):
    def __init__(self, func: Callable, name: str, description: str, parameters: Dict[str, Any]):
        super().__init__(name, description, parameters)
        self.func = func

    def run(self, invocation_state: Dict[str, Any] = None, **kwargs) -> Any:
        # Function tools might not support state, or we pass it if the func accepts it
        # For simplicity, we just pass kwargs here unless the func expects state
        return self.func(**kwargs)
