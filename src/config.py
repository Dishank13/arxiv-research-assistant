"""
Centralized configuration -- all secrets and settings in one place.
"""
import os
import sys
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    tavily_api_key: str = ""
    hf_token: str = ""
    max_query_length: int = 500
    rate_limit_per_minute: int = 20

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """Load and validate settings. Exit with clear message if keys are missing."""
    settings = Settings()
    required = {
        "GROQ_API_KEY": settings.groq_api_key,
        "QDRANT_URL": settings.qdrant_url,
        "QDRANT_API_KEY": settings.qdrant_api_key,
        "TAVILY_API_KEY": settings.tavily_api_key,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
        print("Please set them in your .env file. See .env.example for reference.")
        sys.exit(1)
    return settings


settings = get_settings()
