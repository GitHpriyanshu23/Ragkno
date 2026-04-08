# RAG2: Full-Stack Retrieval-Augmented Generation Platform

RAG2 is a production-style RAG application that supports document ingestion, hybrid retrieval, conversational memory, streaming responses, and source-grounded answers with citations.

It includes:

- FastAPI backend
- React + Vite frontend
- FAISS vector store with metadata persistence
- Hybrid retrieval (dense + BM25)
- Semantic chunking + parent-child chunking
- Google Drive OAuth ingestion
- Server-sent events (token streaming)
- RAGAS evaluation pipeline

## Features

- Multi-source ingestion
	- Upload local files (`.pdf`, `.txt`, `.docx`)
	- Ingest website URLs
	- Sync selected Google Drive files
- Retrieval and generation
	- Dense retrieval with FAISS embeddings
	- Sparse retrieval with BM25
	- Hybrid fusion using reciprocal-rank style merging
	- Semantic reranking (CrossEncoder)
	- Parent-child context expansion before LLM generation
- Chat UX
	- Multi-thread chat history
	- Streaming token responses (SSE)
	- Inline citations and source cards
	- Expandable source chunks and relevance display
- Memory
	- Session-aware memory in SQLite
	- Lightweight summary refresh for longer chats
- Evaluation
	- RAGAS metrics: faithfulness, context precision, answer relevancy

## Tech Stack

- Backend: FastAPI, Uvicorn
- Frontend: React, Vite, React Router
- Retrieval: FAISS, sentence-transformers, rank-bm25
- LLM: Google GenAI chat models
- Evaluation: ragas, datasets
- Storage: local FAISS index + SQLite chat memory

## Project Structure

```text
backend/               FastAPI routes and API orchestration
frontend/              React app (Vite)
src/                   RAG core modules (ingest, retrieval, memory, vectorstore)
evaluation/            RAGAS evaluation scripts and datasets
faiss_store/           Persistent FAISS index and metadata
data/                  Local data and runtime artifacts
```

## Prerequisites

- Python 3.13+
- Node.js 18+
- npm

## Environment Variables

Create a `.env` file in the repository root:

```env
# Required for LLM responses
GOOGLE_API_KEY=your_google_api_key

# Optional: override default model
GOOGLE_LLM_MODEL=gemma-3-12b-it

# Optional: Drive OAuth support
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/callback
```

## Installation

### 1) Python dependencies

From project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

Or if you use `uv`:

```bash
uv sync
```

### 2) Frontend dependencies

```bash
cd frontend
npm install
```

## Running the App

Open two terminals.

### Terminal A: Backend

From project root:

```bash
source .venv/bin/activate
python backend/main.py
```

Backend runs on: `http://localhost:8000`

### Terminal B: Frontend

From project root:

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Frontend runs on: `http://localhost:5173`

## API Overview

- Health
	- `GET /health`
- Query
	- `POST /query`
	- `POST /query/stream` (SSE)
- Memory
	- `POST /memory/reset`
- Ingestion
	- `POST /ingest/files`
	- `POST /ingest/url`
	- `GET /ingest/sources`
	- `POST /ingest/unindex`
- Google Drive
	- `GET /auth/url`
	- `GET /auth/callback`
	- `GET /auth/status`
	- `DELETE /drive/disconnect`
	- `GET /drive/files`
	- `POST /drive/sync`

## Evaluation (RAGAS)

Run evaluation on a JSON test set:

```bash
source .venv/bin/activate
python evaluation/ragas_eval.py --dataset evaluation/testset.sample.json --top-k 3
```

Expected input format (`question`, `ground_truth`):

```json
[
	{
		"question": "What optimization work did Barkha Navalani do?",
		"ground_truth": "She worked on real-world projects involving model training and performance optimization."
	}
]
```

## Notes for First Run

- If you changed chunking/retrieval logic, re-index documents for best results.
- If Drive OAuth is enabled, the app stores Drive tokens in `token.json`.
- Local chat history is stored in `data/chat_memory.sqlite3`.

## Current Retrieval Design

- Semantic chunking for better meaning boundaries
- Parent-child chunk strategy:
	- Retrieve smaller child chunks
	- Expand to larger parent context for generation
- Hybrid dense+sparse retrieval:
	- FAISS dense candidates
	- BM25 sparse candidates
	- Score fusion and reranking

## License

This project is for educational and development use. Add your preferred license before public distribution.
