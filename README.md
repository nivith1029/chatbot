# 📚 RAG PDF Assistant (Ollama + FastAPI + FAISS)

A local Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and ask questions using a locally running Large Language Model (LLM) via Ollama.

This project demonstrates how to build a production-style GenAI document assistant without using paid APIs.

---

## 🚀 Features

- Upload and index PDF documents
- Generate embeddings using Ollama
- Vector storage using FAISS
- Semantic search over documents
- Question answering with citations
- Filename-based document filtering
- Local LLM inference
- REST API with FastAPI
- GitHub Actions CI
- Ruff linting
- Pytest testing

---

## 🏗️ Architecture

Client → FastAPI → FAISS → Ollama (Embeddings + LLM)

---

## 🧰 Tech Stack

- Python 3.10+
- FastAPI
- Ollama
- FAISS
- NumPy
- PyPDF
- Requests
- Pytest
- Ruff

---

## ⚙️ Prerequisites

### Install Ollama

Download:

https://ollama.com

Start Ollama:

ollama serve

Pull models:

ollama pull qwen2.5:1.5b  
ollama pull nomic-embed-text

---

## 📦 Installation

### Clone Repository

git clone https://github.com/nivith1029/chatbot.git  
cd chatbot

### Create Virtual Environment

python3 -m venv .venv  
source .venv/bin/activate

### Install Dependencies

pip install -r requirements.txt

---

## ▶️ Run Server

uvicorn rag_service:app --reload --port 8001

Server URL:

http://127.0.0.1:8001

---

## 📤 Upload PDF

curl -X POST "http://127.0.0.1:8001/rag/ingest" \
-F "file=@document.pdf"

---

## ❓ Query Documents

curl -X POST "http://127.0.0.1:8001/rag/query" \
-H "Content-Type: application/json" \
-d '{
  "question": "What is the deadline?",
  "filename": "document.pdf",
  "top_k": 5
}'

---

## 📌 Example Response

{
  "answer": "- Deadline: April 15, 2025 (Source 1)",
  "sources": [
    {
      "filename": "policy.pdf",
      "page_num": 1
    }
  ],
  "latency_ms": 28000,
  "model": "qwen2.5:1.5b"
}

---

## 🧪 Testing & Linting

Run tests:

pytest

Run lint:

ruff check .

---

## 💼 Use Cases

- HR document assistants
- Legal document search
- Compliance monitoring
- Internal knowledge base
- Contract analysis
- Resume screening
- Research assistants

---

## 📈 Future Improvements

- Web UI
- Docker support
- Cloud deployment
- Authentication
- Caching
- Multi-user support

---

## 👤 Author

Nivith Avula

GitHub: https://github.com/nivith1029

Focus Areas:
- Generative AI
- RAG Systems
- LLM Engineering
- Backend APIs
- Cloud & DevOps

---

## 📄 License

MIT License
