from fastapi import FastAPI
from app.routers import ingest, chat

app = FastAPI(title="YouTube Transcript Chatbot")

app.include_router(ingest.router, prefix="/ingest", tags=["Ingest"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
