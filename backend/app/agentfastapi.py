from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = "llama-3.1-8b-instant"

model = ChatGroq(
    model=MODEL_NAME,
    temperature=0
)

tools = []

agent = create_react_agent(model, tools)

def ask(question: str):

    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })

    return response["messages"][-1].content