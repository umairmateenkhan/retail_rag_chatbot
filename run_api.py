# run_api.py
from fastapi import FastAPI
from app.api import router
from dotenv import load_dotenv
import os

load_dotenv()  # This loads variables from .env into environment

# Ensure the OpenAI API key is set
app = FastAPI(title="Retail RAG Chatbot API")


#define a new HTTP GET endpoint at the root URl
@app.get("/")

#The function root() is defined as the handler for the root endpoint of your FastAPI application. 
# When a client sends an HTTP GET request to the root URL (/), FastAPI will call this function.
def root():
    return {"message": "Welcome to the Retail RAG Chatbot API!"}

#By calling include_router, you are telling FastAPI to add all the routes defined in 
# the router to your main application.
app.include_router(router)
