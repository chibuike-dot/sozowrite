from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title="AI Aggregator API",
    description="Aggregate responses from multiple AI models",
    version="0.1.0"
)

# CORS configuration (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data models
class AIModelResponse(BaseModel):
    model_name: str
    response: str
    confidence: float

class AggregationRequest(BaseModel):
    prompt: str
    models: Optional[List[str]] = ["gpt-4", "claude-3", "gemini-1.5"]

# Root endpoint
@app.get("/")
async def root():
    return {"message": "AI Aggregator API - Send a POST to /aggregate"}

# Aggregation endpoint
@app.post("/aggregate", response_model=List[AIModelResponse])
async def aggregate(request: AggregationRequest):
    """
    Aggregate responses from multiple AI models.
    Mock implementation - replace with actual API calls.
    """
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # Mock responses (replace with real API calls)
    mock_responses = [
        {
            "model_name": "gpt-4",
            "response": f"GPT-4 response to: {request.prompt}",
            "confidence": 0.95
        },
        {
            "model_name": "claude-3",
            "response": f"Claude-3 response to: {request.prompt}",
            "confidence": 0.92
        }
    ]

    # Filter by requested models if specified
    if request.models:
        mock_responses = [
            r for r in mock_responses
            if r["model_name"] in request.models
        ]

    return mock_responses

# To run locally: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
