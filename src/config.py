import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
    BEDROCK_PROFILE = os.getenv("BEDROCK_PROFILE", "default")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-sonnet")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/research_db")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
