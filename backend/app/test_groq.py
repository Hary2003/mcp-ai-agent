import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool

load_dotenv()

@tool
def fake_tool(query: str) -> str:
    """Fake tool"""
    return "ok"

model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=1
).bind_tools([fake_tool])

print("Invoking model...")
print(model.invoke("Search github for recent commits"))