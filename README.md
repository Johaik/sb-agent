# Research Agent System

This project is a sophisticated multi-agent system designed to perform deep research on a given topic. It utilizes a microservices architecture with a FastAPI backend, Celery workers for asynchronous processing, and a Postgres database with pgvector for storage and retrieval.

## Architecture

- **API (`src/api`)**: FastAPI service handling requests and serving results.
- **Worker (`src/worker`)**: Celery workers executing the research agents (Enricher, Planner, Researcher, Critic, Reporter).
- **Database**: PostgreSQL with `pgvector` extension for storing research chunks and embeddings.
- **Cache/Broker**: Redis used as a Celery broker and backend.

## Prerequisites

- Docker and Docker Compose
- **Tavily API Key**: For web search capability.
- **LLM Provider**:
  - **Amazon Bedrock**: AWS Credentials configured.
  - **OpenRouter**: API Key if using OpenRouter.

## Setup

1.  **Environment Variables**
    Copy `env.example` to `.env` and fill in your details:
    ```bash
    cp env.example .env
    ```

    Ensure you set at least:
    - `TAVILY_API_KEY`
    - `BEDROCK_REGION` and `BEDROCK_PROFILE` (if using AWS)
    - or `OPENROUTER_API_KEY` (if using OpenRouter)

2.  **Start the System**
    Run the complete stack using Docker Compose:
    ```bash
    docker-compose up --build
    ```
    
    This will start:
    - Postgres (Port 5432)
    - Redis (Port 6379)
    - API (Port 8000)
    - Worker

## Usage

### 1. Start a Research Job

Trigger a new research task by sending a POST request to the API.

```bash
curl -X POST "http://localhost:8000/research" \
     -H "Content-Type: application/json" \
     -d '{"idea": "The impact of quantum computing on modern cryptography"}'
```

**Response:**
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "pending"
}
```

### 2. Check Job Status

Poll the status of your research job using the returned `job_id`.

```bash
curl "http://localhost:8000/research/YOUR_JOB_ID"
```

The status will progress through:
- `pending`
- `processing` (Enriching and Planning)
- `generating` (Researching and Reporting)
- `completed`

### 3. Retrieve Final Report

Once the status is `completed`, the same endpoint will return the full JSON report in the `report` field.

## Development

- **Scripts**: The `scripts/` directory contains helper utilities for debugging:
  - `python scripts/inspect_job.py [JOB_ID]`: View detailed state of a job.
  - `python scripts/trigger_report.py [JOB_ID]`: Manually trigger report generation if stuck.

- **Local Python Setup** (Optional, if not using Docker for dev):
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

## Agents

The system is composed of specialized agents:
1.  **Enricher**: Expands the initial user idea into a detailed description.
2.  **Planner**: Breaks down the description into actionable research tasks.
3.  **Researcher**: Performs the actual research using Tavily (Web) and RAG (Internal DB).
4.  **Critic**: Reviews research results for quality and relevance.
5.  **Reporter**: Aggregates all findings into a final comprehensive report.
