from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import redis.asyncio as redis
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
import httpx
import logging
from contextlib import asynccontextmanager
from enum import Enum
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import os
from pydantic import ValidationError

from database import (
    init_db,
    log_usage,
    get_api_key_usage,
    get_model_priority,
    get_model_cost,
    get_fallback_chain
)
from models import (
    AIModelResponse,
    AggregationRequest,
    AggregationResponse,
    ProviderConfig,
    UsageLog,
    ModelPriority,
    ErrorResponse
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Provider(str, Enum):
    MISTRAL = "mistral"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"

# Load API keys from environment
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Provider configurations
PROVIDER_CONFIGS: Dict[Provider, ProviderConfig] = {
    Provider.MISTRAL: ProviderConfig(
        base_url="https://api.mistral.ai/v1",
        api_key=MISTRAL_API_KEY,
        api_key_env="MISTRAL_API_KEY",
        free_tier_limit=1000,
        cost_per_token=0.000002,
        priority=ModelPriority.FREE,
        models=["mistral-tiny", "mistral-small"]
    ),
    Provider.GEMINI: ProviderConfig(
        base_url="https://generativelanguage.googleapis.com/v1",
        api_key=GEMINI_API_KEY,
        api_key_env="GEMINI_API_KEY",
        free_tier_limit=500,
        cost_per_token=0.0000025,
        priority=ModelPriority.PAID,
        models=["gemini-1.5-flash", "gemini-1.5-pro"]
    ),
    Provider.DEEPSEEK: ProviderConfig(
        base_url="https://api.deepseek.com/v1",
        api_key=DEEPSEEK_API_KEY,
        api_key_env="DEEPSEEK_API_KEY",
        free_tier_limit=2000,
        cost_per_token=0.0000015,
        priority=ModelPriority.FREE,
        models=["deepseek-chat", "deepseek-coder"]
    )
}

limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database
    await init_db()

    # Initialize rate limiter
    redis_connection = redis.from_url("redis://localhost")

    yield

    # Cleanup
    await redis_connection.close()

app = FastAPI(lifespan=lifespan,
              title="AI Aggregator API",
              description="Priority-based AI model aggregator with fallback",
              version="1.0.0")

# Add rate limiting middleware
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing"
        )
    return api_key

@app.get("/")
async def root():
    return {
        "message": "AI Aggregator API",
        "providers": list(PROVIDER_CONFIGS.keys()),
        "status": "operational"
    }

async def call_provider(
    provider: Provider,
    model_name: str,
    prompt: str,
    max_tokens: int,
    api_key: str
) -> Optional[AIModelResponse]:
    """Helper function to call a single provider with proper error handling"""
    config = PROVIDER_CONFIGS.get(provider)
    if not config or not config.api_key:
        logger.error(f"Provider {provider} not configured properly")
        return None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }

            # Special handling for Gemini API format
            if provider == Provider.GEMINI:
                payload = {
                    "model": f"models/{model_name}",
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": max_tokens
                    }
                }
                endpoint = f"{config.base_url}/models/{model_name}:generateContent"
            else:
                endpoint = f"{config.base_url}/chat/completions"

            response = await client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()

            response_data = response.json()

            # Handle different response formats
            if provider == Provider.GEMINI:
                model_response = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                tokens_used = len(model_response.split())  # Simple approximation
            else:
                model_response = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                tokens_used = response_data.get("usage", {}).get("total_tokens", 0)

            if not model_response:
                logger.error(f"No response content from {provider}-{model_name}")
                return None

            cost = tokens_used * config.cost_per_token

            return AIModelResponse(
                model_name=f"{provider.value}-{model_name}",
                response=model_response,
                confidence=0.95,
                tokens_used=tokens_used,
                cost=cost,
                provider=provider.value
            )

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from {provider}-{model_name}: {str(e)} - {e.response.text if hasattr(e, 'response') else 'No response'}")
        return None
    except (httpx.RequestError, ValidationError, KeyError) as e:
        logger.error(f"Error processing {provider}-{model_name} response: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error with {provider}-{model_name}: {str(e)}", exc_info=True)
        return None

@app.post("/aggregate",
          response_model=AggregationResponse)
@limiter.limit("10/minute")
async def aggregate(
    request: AggregationRequest,
    api_key: str = Depends(get_api_key),
    fastapi_request: Request = None
):
    """
    Aggregate responses from multiple AI models with priority-based routing and fallback.
    """
    # Validate request
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # Get usage stats for this API key
    usage = await get_api_key_usage(api_key)

    # Get priority-ordered list of models with fallback chain
    prioritized_models = await get_model_priority()
    fallback_chain = await get_fallback_chain()

    results = []
    errors = []
    total_tokens = 0
    total_cost = 0.0

    # Try models in priority order
    for model_info in prioritized_models:
        provider_name = model_info['provider']
        model_name = model_info['model_name']

        try:
            provider = Provider(provider_name)
        except ValueError:
            logger.error(f"Invalid provider name: {provider_name}")
            continue

        config = PROVIDER_CONFIGS.get(provider)
        if not config:
            errors.append({
                "provider": provider_name,
                "model": model_name,
                "error": "Provider not configured"
            })
            continue

        # Check rate limits and quotas
        provider_usage = usage.get("usage_by_provider", {}).get(provider_name, {})
        monthly_usage = provider_usage.get("monthly_tokens", 0)

        if monthly_usage >= config.free_tier_limit and config.priority == ModelPriority.FREE:
            logger.warning(f"Free tier limit reached for {provider}, skipping")
            continue

        # Call the provider
        result = await call_provider(
            provider=provider,
            model_name=model_name,
            prompt=request.prompt,
            max_tokens=request.max_tokens or 1000,
            api_key=api_key
        )

        if result:
            results.append(result)
            total_tokens += result.tokens_used
            total_cost += result.cost

            # Log successful usage
            await log_usage(
                api_key=api_key,
                provider=provider_name,
                model_name=model_name,
                tokens_used=result.tokens_used,
                cost=result.cost,
                timestamp=datetime.utcnow()
            )

            # If we got a successful response and don't need all models, we can stop
            if not request.require_all_models and results:
                break
        else:
            errors.append({
                "provider": provider_name,
                "model": model_name,
                "error": "Failed to get response from provider"
            })

    # If no results yet, try fallback chain
    if not results and fallback_chain:
        for fallback in fallback_chain:
            fallback_provider_name = fallback['provider']
            fallback_model = fallback['model_name']

            try:
                fallback_provider = Provider(fallback_provider_name)
            except ValueError:
                logger.error(f"Invalid fallback provider name: {fallback_provider_name}")
                continue

            config = PROVIDER_CONFIGS.get(fallback_provider)
            if not config:
                continue

            # Call the fallback provider
            fallback_result = await call_provider(
                provider=fallback_provider,
                model_name=fallback_model,
                prompt=request.prompt,
                max_tokens=request.max_tokens or 1000,
                api_key=api_key
            )

            if fallback_result:
                fallback_result.is_fallback = True
                results.append(fallback_result)
                total_tokens += fallback_result.tokens_used
                total_cost += fallback_result.cost

                # Log fallback usage
                await log_usage(
                    api_key=api_key,
                    provider=fallback_provider_name,
                    model_name=fallback_model,
                    tokens_used=fallback_result.tokens_used,
                    cost=fallback_result.cost,
                    timestamp=datetime.utcnow()
                )
                break
            else:
                errors.append({
                    "provider": fallback_provider_name,
                    "model": fallback_model,
                    "error": "Fallback provider failed"
                })

    if not results:
        error_details = {
            "message": "All providers failed to return a valid response",
            "errors": errors,
            "suggested_action": "Check provider configurations and API keys"
        }
        logger.error("All providers failed: " + str(error_details))
        raise HTTPException(
            status_code=503,
            detail=error_details
        )

    return AggregationResponse(
        results=results,
        meta={
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "timestamp": datetime.utcnow().isoformat(),
            "errors": errors if errors else None
        }
    )

@app.get("/usage")
async def get_usage(
    api_key: str = Depends(get_api_key),
    provider: Optional[Provider] = None
):
    """
    Get usage statistics for an API key
    """
    usage_data = await get_api_key_usage(api_key, provider.value if provider else None)
    return JSONResponse(content=jsonable_encoder(usage_data))

@app.exception_handler(RateLimitExceeded)
async def rate_limit_exception_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"error": "Too many requests. Please try again later."},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, 'request_id', None)
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
