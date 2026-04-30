from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class ModelPriority(str, Enum):
    FREE = "free"
    PAID = "paid"
    PREMIUM = "premium"

class ProviderConfig(BaseModel):
    base_url: str
    api_key_env: str
    api_key: Optional[str] = None  # Would be set at runtime from env
    free_tier_limit: int
    cost_per_token: float
    priority: ModelPriority
    models: List[str]
    rate_limit: int = 60  # requests per minute

class AIModelResponse(BaseModel):
    model_name: str
    response: str
    confidence: float
    tokens_used: int
    cost: float
    provider: str
    is_fallback: bool = False
    processing_time_ms: Optional[int] = None

class AggregationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    require_all_models: bool = False
    preferred_providers: Optional[List[str]] = None

class AggregationResponse(BaseModel):
    results: List[AIModelResponse]
    meta: Dict[str, Any]

class UsageLog(BaseModel):
    api_key: str
    provider: str
    model_name: str
    tokens_used: int
    cost: float
    timestamp: datetime
    status: str = "completed"

class ModelPriorityConfig(BaseModel):
    provider: str
    model_name: str
    priority: ModelPriority
    cost_per_token: float
    current_load: float = 0.0
    response_time_avg: float = 0.0

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
