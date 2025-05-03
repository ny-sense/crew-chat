# main.py
import time
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
    st= time.time_ns()/ 1000000
    reply = flow.kickoff(inputs={"message": req.message})
    et = time.time_ns() / 1000000
    print(f"time taken: {(et-st)/1000} seconds")
    return ChatResponse(reply=reply)

def plot():
    """Generate a visualization of the flow"""
    flow = ChatbotFlow()
    flow.plot("chatbot_flow")
    print("Flow visualization saved to chatbot_flow.html")