from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.limiter import FastAPILimiter
from fastapi.limiter.depends import RateLimiter
import redis.asyncio as redis
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
import httpx
import logging
from contextlib import asynccontextmanager
from enum import Enum

from .database import (
    init_db,
    log_usage,
    get_api_key_usage,
    get_model_priority,
    get_model_cost,
    get_fallback_chain
)
from .models import (
    AIModelResponse,
    AggregationRequest,
    AggregationResponse,
    ProviderConfig,
    UsageLog,
    ModelPriority
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Provider(str, Enum):
    MISTRAL = "mistral"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"

# Provider configurations (would normally come from secure config)
PROVIDER_CONFIGS: Dict[Provider, ProviderConfig] = {
    Provider.MISTRAL: ProviderConfig(
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        free_tier_limit=1000,
        cost_per_token=0.000002,  # $0.000002 per token
        priority=ModelPriority.FREE,
        models=["mistral-tiny", "mistral-small"]
    ),
    Provider.GEMINI: ProviderConfig(
        base_url="https://generativelanguage.googleapis.com/v1",
        api_key_env="GEMINI_API_KEY",
        free_tier_limit=500,
        cost_per_token=0.0000025,  # $0.0000025 per token
        priority=ModelPriority.PAID,
        models=["gemini-1.5-flash", "gemini-1.5-pro"]
    ),
    Provider.DEEPSEEK: ProviderConfig(
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        free_tier_limit=2000,
        cost_per_token=0.0000015,  # $0.0000015 per token
        priority=ModelPriority.FREE,
        models=["deepseek-chat", "deepseek-coder"]
    )
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database
    await init_db()

    # Initialize rate limiter
    redis_connection = redis.from_url("redis://localhost")
    await FastAPILimiter.init(redis_connection)

    yield

    # Cleanup
    await redis_connection.close()

app = FastAPI(lifespan=lifespan,
              title="AI Aggregator API",
              description="Priority-based AI model aggregator with fallback",
              version="1.0.0")

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

@app.post("/aggregate",
          response_model=AggregationResponse,
          dependencies=[Depends(RateLimiter(times=10, minutes=1))])
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
        provider = model_info['provider']
        model_name = model_info['model_name']
        config = PROVIDER_CONFIGS[provider]

        try:
            # Check rate limits and quotas
            if usage[provider]["monthly_usage"] >= config.free_tier_limit and config.priority == ModelPriority.FREE:
                logger.warning(f"Free tier limit reached for {provider}, skipping")
                continue

            # Make API call
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": request.prompt}],
                    "max_tokens": request.max_tokens or 1000
                }

                response = await client.post(
                    f"{config.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Provider {provider} error: {response.text}"
                    )

                response_data = response.json()
                model_response = response_data["choices"][0]["message"]["content"]
                tokens_used = response_data["usage"]["total_tokens"]

                # Calculate cost
                cost = tokens_used * config.cost_per_token
                total_tokens += tokens_used
                total_cost += cost

                # Log usage
                await log_usage(
                    api_key=api_key,
                    provider=provider,
                    model_name=model_name,
                    tokens_used=tokens_used,
                    cost=cost,
                    timestamp=datetime.utcnow()
                )

                results.append(AIModelResponse(
                    model_name=f"{provider.value}-{model_name}",
                    response=model_response,
                    confidence=0.95,  # Would normally come from response
                    tokens_used=tokens_used,
                    cost=cost,
                    provider=provider.value
                ))

                # If we got a successful response and don't need all models, we can stop
                if not request.require_all_models and results:
                    break

        except Exception as e:
            logger.error(f"Error with {provider}-{model_name}: {str(e)}")
            errors.append({
                "provider": provider.value,
                "model": model_name,
                "error": str(e)
            })
            continue

    if not results and errors:
        # If all providers failed, try fallback chain
        for fallback in fallback_chain:
            fallback_provider = fallback['provider']
            fallback_model = fallback['model_name']
            if fallback_provider in PROVIDER_CONFIGS:
                try:
                    config = PROVIDER_CONFIGS[fallback_provider]
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        headers = {
                            "Authorization": f"Bearer {config.api_key}",
                            "Content-Type": "application/json"
                        }

                        payload = {
                            "model": fallback_model,
                            "messages": [{"role": "user", "content": request.prompt}],
                            "max_tokens": request.max_tokens or 1000
                        }

                        response = await client.post(
                            f"{config.base_url}/chat/completions",
                            json=payload,
                            headers=headers
                        )

                        if response.status_code == 200:
                            response_data = response.json()
                            model_response = response_data["choices"][0]["message"]["content"]
                            tokens_used = response_data["usage"]["total_tokens"]
                            cost = tokens_used * config.cost_per_token

                            await log_usage(
                                api_key=api_key,
                                provider=fallback_provider,
                                model_name=fallback_model,
                                tokens_used=tokens_used,
                                cost=cost,
                                timestamp=datetime.utcnow()
                            )

                            results.append(AIModelResponse(
                                model_name=f"{fallback_provider.value}-{fallback_model}",
                                response=model_response,
                                confidence=0.9,  # Slightly lower for fallback
                                tokens_used=tokens_used,
                                cost=cost,
                                provider=fallback_provider.value,
                                is_fallback=True
                            ))
                            break
                except Exception as e:
                    logger.error(f"Fallback error with {fallback_provider}-{fallback_model}: {str(e)}")
                    continue

    if not results:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "All providers failed",
                "errors": errors,
                "suggested_action": "Try again later or contact support"
            }
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
    usage_data = await get_api_key_usage(api_key, provider)
    return JSONResponse(content=jsonable_encoder(usage_data))

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
            "request_id": request.state.request_id if hasattr(request.state, 'request_id') else None
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
