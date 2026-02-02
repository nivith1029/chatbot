from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import requests
import time
import os
import json
import re
from typing import List, Literal, Optional
from uuid import uuid4
from datetime import datetime

import numpy as np
from pypdf import PdfReader
import faiss

from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

app = FastAPI(title="AI Assistant: Chat + PDF RAG (Ollama + SQLite + FAISS)")

# ===== Ollama =====
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

# ===== DB (SQLite) =====
DB_URL = "sqlite:///./chat.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class ChatMessageDB(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class RagChunkDB(Base):
    __tablename__ = "rag_chunks"
    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(String, index=True)
    filename = Column(String)
    page_num = Column(Integer)
    chunk_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ===== FAISS Store (on disk) =====
RAG_DIR = "./rag_store"
INDEX_PATH = os.path.join(RAG_DIR, "index.faiss")
META_PATH = os.path.join(RAG_DIR, "meta.json")

os.makedirs(RAG_DIR, exist_ok=True)

def _load_meta() -> List[dict]:
    if not os.path.exists(META_PATH):
        return []
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_meta(meta: List[dict]) -> None:
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def _normalize(v: np.ndarray) -> np.ndarray:
    # cosine similarity via inner product after normalization
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norm

def _get_or_create_index(dim: int) -> faiss.Index:
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    # Inner product index (use normalized embeddings)
    return faiss.IndexFlatIP(dim)

def _persist_index(index: faiss.Index) -> None:
    faiss.write_index(index, INDEX_PATH)

# ===== API Models =====
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

class RagIngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_added: int

class RagQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=12)
    temperature: float = Field(default=0.2, ge=0.0, le=1.5)
    max_tokens: int = Field(default=512, ge=1, le=4096)

class RagSource(BaseModel):
    filename: str
    page_num: int
    snippet: str

class RagQueryResponse(BaseModel):
    answer: str
    sources: List[RagSource]
    model: str
    latency_ms: int

# ===== Helpers: DB chat =====
def db_add_chat_message(conversation_id: str, role: str, content: str) -> None:
    db = SessionLocal()
    try:
        db.add(ChatMessageDB(conversation_id=conversation_id, role=role, content=content))
        db.commit()
    finally:
        db.close()

def db_get_chat_messages(conversation_id: str) -> List[Message]:
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

# ===== Helpers: DB rag chunks =====
def db_add_rag_chunk(doc_id: str, filename: str, page_num: int, chunk_text: str) -> int:
    db = SessionLocal()
    try:
        row = RagChunkDB(doc_id=doc_id, filename=filename, page_num=page_num, chunk_text=chunk_text)
        db.add(row)
        db.commit()
        db.refresh(row)
        return row.id
    finally:
        db.close()

def db_get_rag_chunk(chunk_id: int) -> RagChunkDB:
    db = SessionLocal()
    try:
        row = db.query(RagChunkDB).filter(RagChunkDB.id == chunk_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Chunk not found")
        return row
    finally:
        db.close()

# ===== Ollama calls =====
def ollama_chat(messages: List[Message], temperature: float, max_tokens: int) -> dict:
    payload = {
        "model": CHAT_MODEL,
        "messages": [m.model_dump() for m in messages],
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": False,
    }
    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()

def ollama_embed(texts: List[str]) -> np.ndarray:
    # Ollama embeddings endpoint is 1 input at a time; loop for simplicity
    vecs = []
    for t in texts:
        r = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": t}, timeout=180)
        r.raise_for_status()
        data = r.json()
        vecs.append(np.array(data["embedding"], dtype=np.float32))
    return np.vstack(vecs)

# ===== Text processing =====
def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 180) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_size)
        chunk = text[i:j]
        chunks.append(chunk)
        if j == len(text):
            break
        i = max(0, j - overlap)
    return chunks

def pdf_to_page_texts(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(io_bytes := bytes(pdf_bytes))
    # pypdf accepts file-like objects; easiest is to use a BytesIO
    from io import BytesIO
    reader = PdfReader(BytesIO(io_bytes))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return pages

# ===== Routes: base health =====
@app.get("/health")
def health():
    return {"status": "ok"}

# ===== Routes: chat memory =====
@app.post("/chat/new")
def new_chat(req: NewChatRequest):
    conversation_id = str(uuid4())
    db_add_chat_message(conversation_id, "system", req.system_prompt)
    return {"conversation_id": conversation_id, "system_prompt": req.system_prompt}

@app.get("/history/{conversation_id}", response_model=HistoryResponse)
def history(conversation_id: str):
    msgs = db_get_chat_messages(conversation_id)
    if not msgs:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conversation_id, "messages": msgs}

@app.post("/chat/{conversation_id}", response_model=ChatResponse)
def chat(conversation_id: str, req: ChatRequest):
    history_msgs = db_get_chat_messages(conversation_id)
    if not history_msgs:
        raise HTTPException(status_code=404, detail="Conversation not found. Create one with /chat/new")

    db_add_chat_message(conversation_id, "user", req.message)
    msgs_for_model = history_msgs + [Message(role="user", content=req.message)]

    start = time.time()
    try:
        data = ollama_chat(msgs_for_model, req.temperature, req.max_tokens)
        reply = (data.get("message") or {}).get("content", "").strip()
        if not reply:
            raise HTTPException(status_code=502, detail="Empty response from model")

        db_add_chat_message(conversation_id, "assistant", reply)
        latency_ms = int((time.time() - start) * 1000)

        return {
            "conversation_id": conversation_id,
            "reply": reply,
            "model": CHAT_MODEL,
            "latency_ms": latency_ms,
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
        }
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama not reachable. Open Ollama app or run: ollama serve")

# ===== Routes: RAG ingest =====
@app.post("/rag/ingest", response_model=RagIngestResponse)
def rag_ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF")

    pdf_bytes = file.file.read()
    doc_id = str(uuid4())
    filename = file.filename

    try:
        pages = pdf_to_page_texts(pdf_bytes)
        all_meta = _load_meta()

        # If index exists, infer dim from it; else dim from first embedding
        dim = None
        index = None

        chunks_added = 0
        new_vectors = []
        new_meta_rows = []

        for page_num, page_text in enumerate(pages, start=1):
            chunks = chunk_text(page_text)
            if not chunks:
                continue

            # Embed chunks
            embeds = ollama_embed(chunks)  # (n, d)
            if dim is None:
                dim = embeds.shape[1]
                index = _get_or_create_index(dim)

            embeds = _normalize(embeds)

            # Save chunks to DB + meta mapping
            for k, chunk in enumerate(chunks):
                chunk_id = db_add_rag_chunk(doc_id, filename, page_num, chunk)
                new_meta_rows.append({"chunk_id": chunk_id})
            new_vectors.append(embeds)
            chunks_added += len(chunks)

        if chunks_added == 0:
            raise HTTPException(status_code=400, detail="No text found in this PDF")

        # Add to FAISS
        vectors = np.vstack(new_vectors).astype(np.float32)
        if index is None:
            dim = vectors.shape[1]
            index = _get_or_create_index(dim)

        index.add(vectors)
        all_meta.extend(new_meta_rows)
        _persist_index(index)
        _save_meta(all_meta)

        return {"doc_id": doc_id, "filename": filename, "chunks_added": chunks_added}

    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama not reachable. Open Ollama app or run: ollama serve")

# ===== Routes: RAG query =====
@app.post("/rag/query", response_model=RagQueryResponse)
def rag_query(req: RagQueryRequest):
    start = time.time()

    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a PDF to /rag/ingest")

    index = faiss.read_index(INDEX_PATH)
    meta = _load_meta()

    # Embed question
    qvec = ollama_embed([req.question]).astype(np.float32)
    qvec = _normalize(qvec)

    top_k = min(req.top_k, 12)
    scores, idxs = index.search(qvec, top_k)

    # Build sources
    sources: List[RagSource] = []
    context_blocks = []
    for rank, faiss_i in enumerate(idxs[0].tolist(), start=1):
        if faiss_i < 0 or faiss_i >= len(meta):
            continue
        chunk_id = meta[faiss_i]["chunk_id"]
        row = db_get_rag_chunk(int(chunk_id))
        snippet = (row.chunk_text[:260] + "…") if len(row.chunk_text) > 260 else row.chunk_text

        sources.append(RagSource(filename=row.filename, page_num=row.page_num, snippet=snippet))
        context_blocks.append(f"[Source {rank}: {row.filename}, page {row.page_num}] {row.chunk_text}")

    context = "\n\n".join(context_blocks)

    system = (
        "You answer using ONLY the provided sources. "
        "If the answer is not in the sources, say you don't know. "
        "Keep the answer clear and practical. "
        "Cite sources like (Source 1), (Source 2)."
    )

    messages = [
        Message(role="system", content=system),
        Message(role="user", content=f"Question: {req.question}\n\nSources:\n{context}"),
    ]

    try:
        data = ollama_chat(messages, req.temperature, req.max_tokens)
        answer = (data.get("message") or {}).get("content", "").strip()
        latency_ms = int((time.time() - start) * 1000)
        return {"answer": answer, "sources": sources, "model": CHAT_MODEL, "latency_ms": latency_ms}
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama not reachable. Open Ollama app or run: ollama serve")
