import os
import argparse
from src.config import Config
from src.llm.factory import get_llm_provider
from src.tools.base import FunctionTool
from src.tools.tavily_tool import TavilyTool
from src.agents.base import Agent

def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

def main():
    parser = argparse.ArgumentParser(description="Run an AI Agent")
    parser.add_argument("--provider", type=str, default="bedrock", choices=["bedrock", "openrouter"], help="LLM Provider to use")
    parser.add_argument("--prompt", type=str, default="Who is the current CEO of Twitter?", help="Initial prompt for the agent")
    args = parser.parse_args()

    print(f"Initializing Agent with {args.provider}...")
    
    try:
        llm = get_llm_provider(args.provider)
    except Exception as e:
        print(f"Failed to initialize LLM provider: {e}")
        return

    # Define tools
    tools = [
        TavilyTool(),
        FunctionTool(
            func=add,
            name="add",
            description="Adds two numbers",
            parameters={
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        ),
        FunctionTool(
            func=multiply,
            name="multiply",
            description="Multiplies two numbers",
            parameters={
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        )
    ]

    agent = Agent(
        name="SearchAgent",
        instructions="You are a helpful assistant. You have access to a web search tool (tavily_search). Use it when asked about current events or information you don't know.",
        llm=llm,
        tools=tools
    )

    print(f"Agent ready. Processing prompt: {args.prompt}")
    response = agent.run(args.prompt)
    print("\nFinal Response:")
    print(response)

if __name__ == "__main__":
    main()
