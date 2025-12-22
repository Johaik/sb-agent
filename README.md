# Multi-Agent Infrastructure

This project provides a flexible infrastructure for building multi-agent systems using Amazon Bedrock and OpenRouter (OpenAI-compatible).

## Setup

1.  **Virtual Environment**:
    The project uses a virtual environment. Ensure you have activated it:
    ```bash
    source .venv/bin/activate
    ```

2.  **Dependencies**:
    Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration**:
    Copy `env.example` to `.env` and fill in your credentials:
    ```bash
    cp env.example .env
    ```
    
    - For Bedrock: Ensure your AWS profile is configured (`aws configure`).
    - For OpenRouter: Set `OPENROUTER_API_KEY`.

## Usage

Run the agent via the CLI:

```bash
# Use Bedrock (Default)
python -m src.main --provider bedrock --prompt "What is 25 * 55?"

# Use OpenRouter
python -m src.main --provider openrouter --prompt "What is 25 * 55?"
```

## Structure

- `src/agents/`: Agent logic and orchestration.
- `src/tools/`: Tool definitions (Calculator, etc.).
- `src/llm/`: LLM Provider implementations (Bedrock, OpenRouter).
- `src/config.py`: Configuration management.

## Features

- **Provider Agnostic**: Switch between Bedrock (Claude 3) and OpenRouter easily.
- **Tool Support**: Agents can execute Python functions as tools.
- **Standardized History**: Handles message history conversion between different provider formats (e.g., Bedrock `user` vs OpenAI `tool` roles).

