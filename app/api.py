# app/api.py
from fastapi import APIRouter
from pydantic import BaseModel
from app.chat_engine import agent
import sqlite3
router = APIRouter()

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    result = agent.run(request.query)
    return ChatResponse(response=result)


def log_chat_to_db(user: str, message: str, answer: str):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user TEXT,
                        message TEXT,
                        answer TEXT
                    )''')
    cursor.execute("INSERT INTO chat_history (user, message, answer) VALUES (?, ?, ?)", (user, message, answer))
    conn.commit()
    conn.close()
