# main.py
from fastapi import FastAPI
from pydantic import BaseModel

from src.chatbot_flow import ChatbotFlow

app = FastAPI()
flow = ChatbotFlow()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply = flow.kickoff(inputs={"message": req.message})
    return ChatResponse(reply=reply)

def plot():
    """Generate a visualization of the flow"""
    flow = ChatbotFlow()
    flow.plot("chatbot_flow")
    print("Flow visualization saved to chatbot_flow.html")