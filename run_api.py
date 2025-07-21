# run_api.py
from fastapi import FastAPI
from app.api import router

app = FastAPI(title="Retail RAG Chatbot API")

@app.get("/")
def root():
    return {"message": "Welcome to the Retail RAG Chatbot API!"}

app.include_router(router)
