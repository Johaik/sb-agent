# Research Agent System

This project is a sophisticated multi-agent system designed to perform deep, comprehensive research on any given topic. It features a rigorous multi-stage research pipeline with quality assurance at every step, utilizing a microservices architecture with FastAPI backend, Celery workers for asynchronous processing, and PostgreSQL with pgvector for intelligent knowledge storage and retrieval.

## Key Features

### Comprehensive Research
- **Multi-stage pipeline**: 9 specialized agents with quality checks
- **Deep web search**: 40-50+ Tavily API calls per research job
- **Knowledge reuse**: RAG system retrieves relevant past research
- **Detailed reports**: 3,000-6,000 word reports with technical depth

### Quality Assurance
- **Evidence scoring**: Every finding rated for relevance and credibility
- **Contradiction seeking**: Actively searches for opposing viewpoints
- **Multi-level review**: Critic agents at task and report levels
- **Feedback loops**: Rejected research gets specific improvement guidance

### Intelligent Knowledge Base
- **Vector storage**: All research stored with embeddings in pgvector
- **Semantic search**: Find relevant past research by meaning, not keywords
- **Cumulative learning**: System gets smarter with each research job
- **Reduced redundancy**: Leverages existing knowledge before new searches

### Production Ready
- **Async processing**: Celery workers for background research
- **Progress tracking**: Real-time status and percentage updates
- **Error handling**: Graceful failures with detailed error messages
- **Optional auth**: Secure API with bearer token authentication
- **Docker deployment**: Complete stack with single command

## Architecture

- **API (`src/api`)**: FastAPI service with optional authentication handling requests and serving results.
- **Worker (`src/worker`)**: Celery workers executing a 9-agent research pipeline with quality checks.
- **Database**: PostgreSQL with `pgvector` extension for vector-based knowledge retrieval (RAG).
- **Cache/Broker**: Redis used as Celery broker and result backend.
- **LLM Providers**: Support for AWS Bedrock and OpenRouter.
- **Research Tools**: Tavily API for web search, internal RAG for knowledge reuse.

## Prerequisites

- Docker and Docker Compose
- **Tavily API Key**: Required for web search capability.
- **LLM Provider** (choose one):
  - **Amazon Bedrock**: AWS Credentials configured (recommended for production).
  - **OpenRouter**: API Key for accessing various LLM models.
- **API Authentication** (optional): Generate a secure token for API access.

## Setup

1.  **Environment Variables**
    Copy `env.example` to `.env` and fill in your details:
    ```bash
    cp env.example .env
    ```

    Required settings:
    - `TAVILY_API_KEY`: Your Tavily API key for web search
    - `BEDROCK_REGION` and `BEDROCK_PROFILE`: If using AWS Bedrock
    - `OPENROUTER_API_KEY`: If using OpenRouter instead
    
    Optional settings:
    - `API_AUTH_ENABLED=true`: Enable API authentication
    - `API_SECRET_KEY`: Your secure API token (generate with `openssl rand -hex 32`)

2.  **Start the System**
    Run the complete stack using Docker Compose:
    ```bash
    docker-compose up --build -d
    ```
    
    This will start:
    - PostgreSQL with pgvector (Port 5432)
    - Redis (Port 6379)
    - FastAPI Server (Port 8000)
    - Celery Worker (background)

## Usage

### 1. Start a Research Job

Trigger a new research task by sending a POST request to the API.

**Without Authentication:**
```bash
curl -X POST "http://localhost:8000/research" \
     -H "Content-Type: application/json" \
     -d '{"idea": "The impact of quantum computing on modern cryptography"}'
```

**With Authentication (if enabled):**
```bash
curl -X POST "http://localhost:8000/research" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_API_SECRET_KEY" \
     -d '{"idea": "The impact of quantum computing on modern cryptography"}'
```

**Response:**
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "progress_percent": 0,
  "current_phase": "queued",
  "created_at": "2025-12-28T19:00:00Z"
}
```

### 2. Check Job Status

Poll the status of your research job using the returned `job_id`.

```bash
curl "http://localhost:8000/research/YOUR_JOB_ID" \
     -H "Authorization: Bearer YOUR_API_SECRET_KEY"
```

The job progresses through multiple phases:
- **enriching**: Expanding and clarifying the research topic
- **planning**: Breaking down into specific research tasks
- **researching**: Executing research tasks with quality checks
- **reporting**: Synthesizing findings into comprehensive report
- **completed**: Final report ready

**Progress tracking:**
- Progress percentage: 0-100%
- Current phase indicator
- Real-time status updates

### 3. Retrieve Final Report

Once `status` is `completed`, the response includes the full research report:

```json
{
  "job_id": "...",
  "status": "completed",
  "progress_percent": 100,
  "current_phase": "reporting",
  "description": "Detailed research description...",
  "report": {
    "summary": "Comprehensive multi-paragraph overview...",
    "key_findings": [
      "Finding 1 with specific details...",
      "Finding 2 with technical specifications..."
    ],
    "details": {
      "Section 1": "In-depth analysis with examples...",
      "Section 2": "Technical specifications and data..."
    }
  }
}
```

**Report Features:**
- Comprehensive summaries (3-4 paragraphs)
- 7-15 detailed key findings
- Multiple detailed sections (200-400+ words each)
- Technical specifications and examples preserved
- Citations and data points included

## Research Pipeline

The system employs a rigorous 9-agent pipeline with quality assurance at every step:

### Stage 1: Preparation
1. **Enricher Agent**: Expands brief ideas into detailed research descriptions
2. **Planner Agent**: Breaks down the topic into 5-10 specific research tasks

### Stage 2: Research Execution (per task)
3. **Hypothesis Generator**: Formulates testable hypotheses for each task
4. **Researcher Agent**: Conducts deep research using:
   - **Tavily API**: Real-time web search with deep search mode
   - **RAG System**: Retrieves relevant past research from vector database
   - Performs 5-10+ searches per task for comprehensive coverage
5. **Evidence Scorer**: Rates relevance (0-10) and credibility (0-10) of findings

### Stage 3: Quality Assurance (per task)
6. **Contradiction Seeker**: Actively searches for opposing views and conflicting data
7. **Critic Agent**: Reviews completeness, relevance, and depth
   - **Approved**: Task moves to final report
   - **Rejected**: Returns to Researcher with specific feedback

### Stage 4: Report Generation
8. **Reporter Agent**: Synthesizes all approved research into structured report
   - Preserves technical details and specifications
   - Generates 3,000-6,000 word comprehensive reports
   - Includes multi-paragraph sections with examples
9. **Final Critic**: Reviews entire report for coherence and quality

### Knowledge Storage
- All completed research is chunked and stored with embeddings in pgvector
- Future research queries leverage past findings via RAG
- Builds institutional knowledge base over time
- Reduces redundant web searches for common topics

**Total Quality Checkpoints**: 3 per task + 1 final review = rigorous accuracy

## Configuration

### Agent Model Configuration

Each agent can be configured with different LLM models for optimal cost/quality trade-offs. See `AGENT_MODEL_RECOMMENDATIONS.md` for details.

**Current Configuration:**
- **Enricher, Planner**: Can use lighter models (e.g., gpt-4o-mini)
- **Researcher, Reporter, Critic**: Use Claude 3.5 Sonnet for best quality
- **Default**: Claude 3 Sonnet via AWS Bedrock

### Token Limits

Agents are configured with appropriate token limits for comprehensive output:
- **ReporterAgent**: 8000 tokens (~6000 words)
- **ResearcherAgent**: 6000 tokens (~4500 words)
- **Other agents**: 2000-4000 tokens

These limits ensure detailed reports without truncation.

### Environment Variables

Key configuration options in `.env`:

```bash
# LLM Provider
BEDROCK_REGION=us-east-1
BEDROCK_PROFILE=default
OPENROUTER_API_KEY=your_key_here

# Search
TAVILY_API_KEY=your_tavily_key

# Database
DATABASE_URL=postgresql://user:password@db:5432/research_db
REDIS_URL=redis://redis:6379/0

# Security
API_AUTH_ENABLED=false
API_SECRET_KEY=your_secure_token_here
```

## Development

### Monitoring & Debugging

- **Check System Health**:
  ```bash
  curl http://localhost:8000/health
  curl http://localhost:8000/ready
  ```

- **View Worker Logs**:
  ```bash
  docker logs sb-agent-worker-1 -f
  ```

- **Inspect Database**:
  ```bash
  docker exec -it sb-agent-db-1 psql -U user -d research_db
  ```

- **Check RAG Storage**:
  ```sql
  -- View stored research chunks
  SELECT COUNT(*), COUNT(DISTINCT job_id) FROM research_chunks;
  
  -- View recent jobs
  SELECT id, status, idea, created_at FROM research_reports 
  ORDER BY created_at DESC LIMIT 10;
  ```

### Scripts

The `scripts/` directory contains helper utilities:
- `python scripts/debug_imports.py`: Test import paths
- `python scripts/test_rag.py`: Test RAG retrieval functionality
- `python scripts/view_trace.py [JOB_ID]`: View detailed execution trace

### Local Development Setup

For development without Docker:
```bash
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

Set environment variables and run services:
```bash
# Start API
uvicorn src.api.main:app --reload

# Start Worker
celery -A src.worker.celery_app worker --loglevel=INFO
```

## Performance & Cost

### Typical Research Job
- **Duration**: 3-5 minutes for comprehensive research
- **API Calls**: 40-50 Tavily searches
- **Output**: 3,000-6,000 word detailed report
- **Storage**: ~35 vector chunks per job

### Cost Optimization
- **RAG first**: System checks internal database before web search
- **Incremental knowledge**: Each job adds to the collective intelligence
- **Configurable depth**: Adjust search depth and task count as needed

## Troubleshooting

### Job Stuck in "enriching" Phase
- Check worker logs: `docker logs sb-agent-worker-1 -f`
- Verify AWS credentials are valid (if using Bedrock)
- Ensure Tavily API key is set correctly

### No Results from RAG
- Verify pgvector extension is installed:
  ```sql
  SELECT * FROM pg_extension WHERE extname = 'vector';
  ```
- Check if chunks are being saved:
  ```sql
  SELECT COUNT(*) FROM research_chunks;
  ```

### API Authentication Errors
- Ensure `API_AUTH_ENABLED` matches your curl commands
- Generate new token: `openssl rand -hex 32`
- Check Authorization header format: `Bearer YOUR_TOKEN`

### Worker Not Processing Tasks
- Restart worker: `docker-compose restart worker`
- Check Redis connection: `docker exec sb-agent-redis-1 redis-cli ping`
- Verify celery can connect: `docker exec sb-agent-worker-1 celery -A src.worker.celery_app inspect ping`

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
