# app/api.py
from fastapi import APIRouter, Header
from pydantic import BaseModel
from app.chat_engine import agent
import sqlite3


router = APIRouter()

# Define request and response models
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

# Define the API endpoint for chat
@router.post("/chat", response_model=ChatResponse)

# Chat endpoint that processes user queries and returns responses   
def chat_endpoint(request: ChatRequest, x_user: str = Header(...)):
    result = agent.run(request.query)
    log_chat_to_db(x_user, request.query, result)
    return ChatResponse(response=result)


# Function to log chat messages to the database
# This function creates a SQLite database to store chat history
# It creates a table if it doesn't exist and inserts the user's message and the assistant's response
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
