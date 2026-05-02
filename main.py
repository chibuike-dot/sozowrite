import os
import httpx
import logging
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/aggregate")
async def aggregate(req: PromptRequest, x_api_key: str = Header(...)):
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": req.prompt}],
                "max_tokens": 1000
            }
        )
        r.raise_for_status()
        data = r.json()
        return {"content": data["choices"][0]["message"]["content"]}

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

os.makedirs("/root/myproject/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/app")
def serve_app():
    return FileResponse("static/index.html")
