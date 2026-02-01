from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests
import time
from typing import List, Literal, Optional
from uuid import uuid4
from datetime import datetime

from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

app = FastAPI(title="AI Chat API (Ollama) + SQLite Memory")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"

# ---- DB (SQLite) ----
DB_URL = "sqlite:///./chat.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class ChatMessageDB(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, index=True)
    role = Column(String)  # system/user/assistant
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ---- API Models ----
Role = Literal["system", "user", "assistant"]

class Message(BaseModel):
    role: Role
    content: str = Field(min_length=1, max_length=8000)

class NewChatRequest(BaseModel):
    system_prompt: str = Field(default="You are a helpful assistant.", min_length=1, max_length=2000)

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    temperature: float = Field(default=0.2, ge=0.0, le=1.5)
    max_tokens: int = Field(default=256, ge=1, le=4096)

class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    model: str
    latency_ms: int
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

class HistoryResponse(BaseModel):
    conversation_id: str
    messages: List[Message]

# ---- Helpers ----
def db_add_message(conversation_id: str, role: str, content: str):
    db = SessionLocal()
    try:
        db.add(ChatMessageDB(conversation_id=conversation_id, role=role, content=content))
        db.commit()
    finally:
        db.close()

def db_get_messages(conversation_id: str) -> List[Message]:
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMessageDB)
            .filter(ChatMessageDB.conversation_id == conversation_id)
            .order_by(ChatMessageDB.created_at.asc(), ChatMessageDB.id.asc())
            .all()
        )
        return [Message(role=row.role, content=row.content) for row in rows]
    finally:
        db.close()

def call_ollama(messages: List[Message], temperature: float, max_tokens: int):
    payload = {
        "model": MODEL_NAME,
        "messages": [m.model_dump() for m in messages],
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": False,
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

# ---- Routes ----
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat/new")
def new_chat(req: NewChatRequest):
    conversation_id = str(uuid4())
    db_add_message(conversation_id, "system", req.system_prompt)
    return {"conversation_id": conversation_id, "system_prompt": req.system_prompt}

@app.get("/history/{conversation_id}", response_model=HistoryResponse)
def history(conversation_id: str):
    msgs = db_get_messages(conversation_id)
    if not msgs:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conversation_id, "messages": msgs}

@app.post("/chat/{conversation_id}", response_model=ChatResponse)
def chat(conversation_id: str, req: ChatRequest):
    # Load history
    history_msgs = db_get_messages(conversation_id)
    if not history_msgs:
        raise HTTPException(status_code=404, detail="Conversation not found. Create one with /chat/new")

    # Add user message to DB
    db_add_message(conversation_id, "user", req.message)

    # Build message list for the model (history + new user msg)
    msgs_for_model = history_msgs + [Message(role="user", content=req.message)]

    start = time.time()
    try:
        data = call_ollama(msgs_for_model, req.temperature, req.max_tokens)
        reply = (data.get("message") or {}).get("content", "").strip()
        if not reply:
            raise HTTPException(status_code=502, detail="Empty response from model")

        # Save assistant reply
        db_add_message(conversation_id, "assistant", reply)

        latency_ms = int((time.time() - start) * 1000)
        return {
            "conversation_id": conversation_id,
            "reply": reply,
            "model": MODEL_NAME,
            "latency_ms": latency_ms,
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
        }

    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama not reachable. Open Ollama app or run: ollama serve")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Model timed out")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")
