# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chat AXE is a RAG (Retrieval-Augmented Generation) chatbot with a Flask web UI. Users upload documents (PDF, PPTX) or scrape web pages into named "collections," which are chunked, embedded, and stored in PostgreSQL with pgvector. At query time, the most relevant collection is selected automatically, context chunks are retrieved, and passed to Groq's LLM API for a response.

## Running Locally

```
# Activate venv
.venv\Scripts\activate

# Run the web app (serves on http://localhost:5000)
python app.py
```

No Docker required — the app uses a cloud Neon PostgreSQL database and HuggingFace Inference API for embeddings, both configured via `.env`.

## Architecture

**Two entry points:**
- `app.py` — Flask web server (primary). Serves the single-page UI and all `/api/*` endpoints.
- `chatbot.py` — Legacy CLI chatbot. Uses hardcoded Ollama embeddings and local DB. Not used in production.

**Core module:**
- `knowledgebase.py` — All data layer logic: document readers, vector store (`PgVector2`), knowledge base classes, multi-collection routing, and collection CRUD (backed by a `_collections` table in PostgreSQL).

**Request flow:**
1. `POST /api/chat` → `get_knowledge_base()` loads all collection names from `_collections` table → `MultiKnowledgeBase.get_context()` finds the best collection by comparing query embedding distance across all collections → retrieves top-k chunks → builds prompt with context → calls Groq API → returns response.

**Ingestion flow:**
1. Upload/URL → spawns background `threading.Thread` → reader chunks the document → each chunk is embedded via `get_embedding()` → stored in a per-collection pgvector table → collection name added to `_collections` table.

**Embedding providers (controlled by `EMBEDDING_PROVIDER` env var):**
- `huggingface` (cloud deployment) — calls `router.huggingface.co/hf-inference/models/{model}`. Uses `BAAI/bge-small-en-v1.5` (384d).
- `ollama` (local development) — calls local Ollama API. Uses `nomic-embed-text:latest` (768d).

**Key constraint:** `EMBEDDING_DIM` must match the model's output dimension. Mixing dimensions (e.g., ingesting with 768d Ollama then querying with 384d HuggingFace) will cause pgvector errors.

## Environment Variables

All config is in `.env` (loaded by `python-dotenv`). Key variables:
- `DATABASE_URL` — PostgreSQL connection string (must use `postgresql+psycopg2://` prefix for SQLAlchemy)
- `GROQ_API_KEY` — required for LLM responses
- `EMBEDDING_PROVIDER` — `huggingface` or `ollama`
- `EMBEDDING_MODEL` / `EMBEDDING_DIM` — must be consistent across ingestion and queries
- `HF_API_TOKEN` — HuggingFace token (optional but recommended for rate limits)

## Deployment

Deployed on Render.com. Config in `render.yaml` and `Procfile`.
- Production uses gunicorn: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
- Database: Neon PostgreSQL (free tier with pgvector)
- Set `PYTHON_VERSION=3.10.12` in Render environment to avoid Python 3.14 compatibility issues

## Important Patterns

- Collection names are used directly as PostgreSQL table names. `validate_collection_name()` enforces `^[a-zA-Z_][a-zA-Z0-9_]*$` — always validate before any DB operation.
- `PgVector2` creates its table on instantiation (`_ensure_table`), so constructing the object has side effects.
- Collection tracking uses the `_collections` PostgreSQL table (not the filesystem), so it persists across deploys.
- The HuggingFace embedding response format varies by model — `_embed_huggingface()` handles flat arrays, nested arrays, and token-level embeddings with mean pooling.
