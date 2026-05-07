from fastapi import FastAPI
from pydantic import BaseModel

from agentfastapi import ask

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {
        "message": "AI Agent Backend Running"
    }


@app.post("/chat")
def chat(req: ChatRequest):

    answer = ask(req.message)

    return {
        "answer": answer
    }