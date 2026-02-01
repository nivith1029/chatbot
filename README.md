# AI Chat API (Ollama + FastAPI + SQLite Memory)

A local, production-style LLM chat backend that supports conversation memory using SQLite.

## Features
- Local LLM inference via Ollama
- FastAPI REST endpoints
- Conversation sessions (conversation_id)
- Persistent chat history (SQLite)
- Latency + token stats

## Requirements
- Python 3.11+
- Ollama installed and running
- Model pulled: llama3.2:3b

## Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run
uvicorn main:app --reload

## Endpoints
- POST /chat/new
- POST /chat/{conversation_id}
- GET  /history/{conversation_id}
- GET  /health

## Example
Create chat:
curl -X POST http://127.0.0.1:8000/chat/new -H "Content-Type: application/json" -d '{"system_prompt":"You are helpful."}'

Chat:
curl -X POST http://127.0.0.1:8000/chat/<ID> -H "Content-Type: application/json" -d '{"message":"Hello"}'

History:
curl http://127.0.0.1:8000/history/<ID>
