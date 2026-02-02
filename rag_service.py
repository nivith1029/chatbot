from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import os, json, time
from typing import List, Optional
import numpy as np
import requests
import faiss
from pypdf import PdfReader
from io import BytesIO

app = FastAPI(title="RAG PDF Assistant (Ollama + FAISS)")

# Ollama endpoints
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

# Models
CHAT_MODEL = "qwen2.5:1.5b"
EMBED_MODEL = "nomic-embed-text"

# Storage
RAG_DIR = "./rag_store"
INDEX_PATH = os.path.join(RAG_DIR, "index.faiss")
META_PATH = os.path.join(RAG_DIR, "meta.json")
os.makedirs(RAG_DIR, exist_ok=True)


def load_meta():
    if not os.path.exists(META_PATH):
        return []
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_meta(meta):
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


def get_or_create_index(dim: int) -> faiss.Index:
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return faiss.IndexFlatIP(dim)


def persist_index(index: faiss.Index):
    faiss.write_index(index, INDEX_PATH)


def clean_text(s: str) -> str:
    s = (s or "").replace("\x00", " ")
    return " ".join(s.split()).strip()


def chunk_text(text: str, chunk_size=900, overlap=180):
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_size)
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return chunks


def pdf_pages(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(BytesIO(pdf_bytes))
    return [(p.extract_text() or "") for p in reader.pages]


def ollama_embed(texts: List[str]) -> np.ndarray:
    vecs = []
    for t in texts:
        r = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": t},
            timeout=180,
        )
        r.raise_for_status()
        vecs.append(np.array(r.json()["embedding"], dtype=np.float32))
    return np.vstack(vecs)


def ollama_chat(user_prompt: str, temperature: float, max_tokens: int = 256) -> str:
    # Keep this prompt simple so small models behave
    system = (
        "You are a document assistant.\n"
        "Use ONLY the facts provided in the USER message under CONTEXT.\n"
        "Do NOT mention the word 'context'.\n"
        "If the answer is missing, say exactly: I don't know based on the document.\n"
        "Return bullet points only.\n"
    )

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": False,
    }

    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=180)
    r.raise_for_status()
    return (r.json().get("message") or {}).get("content", "").strip()


class QueryReq(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=12)
    filename: Optional[str] = Field(default=None, max_length=300)  # optional filter
    temperature: float = Field(default=0.0, ge=0.0, le=1.5)
    max_tokens: int = Field(default=256, ge=1, le=2048)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rag/ingest")
def ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF file")

    pdf_bytes = file.file.read()
    pages = pdf_pages(pdf_bytes)
    meta = load_meta()

    index = None
    dim = None
    added = 0

    for page_num, page_text in enumerate(pages, start=1):
        chunks = chunk_text(page_text)
        if not chunks:
            continue

        embeds = ollama_embed(chunks)
        if dim is None:
            dim = embeds.shape[1]
            index = get_or_create_index(dim)

        embeds = normalize(embeds).astype(np.float32)

        index.add(embeds)
        for c in chunks:
            meta.append({"filename": file.filename, "page_num": page_num, "text": c})
        added += len(chunks)

    if added == 0:
        raise HTTPException(status_code=400, detail="No text found in PDF")

    persist_index(index)
    save_meta(meta)
    return {"filename": file.filename, "chunks_added": added}


@app.post("/rag/query")
def query(req: QueryReq):
    start = time.time()

    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise HTTPException(status_code=400, detail="No docs indexed yet. Upload a PDF to /rag/ingest")

    index = faiss.read_index(INDEX_PATH)
    meta = load_meta()

    qv = normalize(ollama_embed([req.question]).astype(np.float32))
    _, idxs = index.search(qv, min(req.top_k, 12))

    sources = []
    context_blocks = []

    rank = 0
    seen = set()  # dedupe repeated chunks

    for i in idxs[0].tolist():
        if i < 0 or i >= len(meta):
            continue

        m = meta[i]

        if req.filename and m["filename"] != req.filename:
            continue

        key = (m["filename"], m["page_num"], m["text"][:200])
        if key in seen:
            continue
        seen.add(key)

        rank += 1
        snippet = m["text"][:260] + ("…" if len(m["text"]) > 260 else "")
        sources.append({"source": rank, "filename": m["filename"], "page_num": m["page_num"], "snippet": snippet})
        context_blocks.append(f"[Source {rank}: {m['filename']}, page {m['page_num']}] {m['text']}")

        if rank >= min(req.top_k, 12):
            break

    if not sources:
        raise HTTPException(status_code=404, detail="No matching sources found for that query/filter")

    context = "\n\n".join(context_blocks)

    # Keep prompt small for speed
    MAX_CONTEXT_CHARS = 3500
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[trimmed]"

    prompt = (
        "Answer the QUESTION using ONLY the facts in CONTEXT.\n"
        "Return up to 3 bullet points (1 to 3). Only include bullets that directly answer the question.\n"
        "Each bullet must include a citation like (Source 1).\n"
        "\n"
        f"QUESTION:\n{req.question}\n\n"
        f"CONTEXT:\n{context}\n"
    )

    answer = ollama_chat(prompt, req.temperature, req.max_tokens)

    # Force citation formatting when only 1 source exists
    if len(sources) == 1:
        lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
        fixed = []
        for ln in lines:
            if not ln.startswith(("-", "•")):
                ln = f"- {ln}"
            if "(Source 1)" not in ln:
                ln = f"{ln} (Source 1)"
            fixed.append(ln)
        answer = "\n".join(fixed[:3]) if fixed else "- I don't know based on the document. (Source 1)"

    # Basic quality check: if answer has no doc keywords, regenerate once
    doc_text = " ".join([s["snippet"] for s in sources]).lower()
    must_have = ["work", "leave", "confidential", "april", "effective", "hours", "days", "deadline"]
    if not any(w in answer.lower() for w in must_have) and any(w in doc_text for w in must_have):
        stronger = (
            prompt
            + "\nIMPORTANT: Use facts from the document like dates, work hours, leave rules, confidentiality, and deadlines."
        )
        answer2 = ollama_chat(stronger, 0.0, 180)
        if answer2.strip():
            answer = answer2
            if len(sources) == 1:
                lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
                fixed = []
                for ln in lines:
                    if not ln.startswith(("-", "•")):
                        ln = f"- {ln}"
                    if "(Source 1)" not in ln:
                        ln = f"{ln} (Source 1)"
                    fixed.append(ln)
                answer = "\n".join(fixed[:3]) if fixed else "- I don't know based on the document. (Source 1)"

    latency_ms = int((time.time() - start) * 1000)
    return {"answer": answer, "sources": sources, "latency_ms": latency_ms, "model": CHAT_MODEL}
