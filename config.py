from pydantic import BaseSettings, Field
from typing import Optional, Dict, Any
import logging

class Settings(BaseSettings):
    # API Keys (would normally come from secure vault)
    mistral_api_key: str = Field(..., env="MISTRAL_API_KEY")
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    deepseek_api_key: str = Field(..., env="DEEPSEEK_API_KEY")

    # Rate limiting
    rate_limit_requests: int = Field(10, env="RATE_LIMIT_REQUESTS")
    rate_limit_minutes: int = Field(1, env="RATE_LIMIT_MINUTES")

    # Database
    database_url: str = Field("ai_aggregator.db", env="DATABASE_URL")

    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Provider configurations (can be overridden by env vars)
    provider_configs: Dict[str, Dict[str, Any]] = {
        "mistral": {
            "free_tier_limit": 1000,
            "cost_per_token": 0.000002
        },
        "gemini": {
            "free_tier_limit": 500,
            "cost_per_token": 0.0000025
        },
        "deepseek": {
            "free_tier_limit": 2000,
            "cost_per_token": 0.0000015
        }
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Initialize settings
settings = Settings()

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Validate API keys are set
if not settings.mistral_api_key:
    logger.warning("MISTRAL_API_KEY not set - Mistral provider will not work")
if not settings.gemini_api_key:
    logger.warning("GEMINI_API_KEY not set - Gemini provider will not work")
if not settings.deepseek_api_key:
    logger.warning("DEEPSEEK_API_KEY not set - DeepSeek provider will not work")
