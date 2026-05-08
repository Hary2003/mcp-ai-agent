# backend/app/main.py
# Only responsibility: HTTP server.
# No business logic or agent orchestration should live here.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from app.agent.agent import ask

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI(
    title="MCP Agent API",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class ChatRequest(BaseModel):
    message: str

# Response schema
class ChatResponse(BaseModel):
    response: str

# Health check route
@app.get("/")
async def root():
    return {
        "status": "MCP Agent API running"
    }

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):

    message = body.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    try:
        answer = await ask(
            message,
            verbose=False
        )

        return ChatResponse(
            response=answer
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )