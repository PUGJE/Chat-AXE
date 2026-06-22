from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import re
import threading
import uuid
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from knowledgebase import (
    MultiKnowledgeBase, PgVector2, load_collections, add_collection,
    remove_collection, validate_collection_name, init_collections_table,
    SimplePDFReader, PDFKnowledgeBase, WebReader, WebKnowledgeBase,
    PPTXReader, PPTXKnowledgeBase
)
import requests

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://cu:cu@127.0.0.1:5532/cu")
init_collections_table(DB_URL)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

ingestion_jobs = {}


class Groq:
    def __init__(self, model_id, api_key=None):
        self.model_id = model_id
        self.api_key = api_key or GROQ_API_KEY
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def complete(self, prompt, history=None):
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Use the provided context if relevant, but always answer the user's question directly using your own knowledge too. Never say 'the context does not contain' or mention lack of context — just answer naturally."},
            *([{"role": h["role"], "content": h["content"]} for h in (history or [])]),
            {"role": "user", "content": prompt}
        ]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"model": self.model_id, "messages": messages}
        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def get_embedding(text_input):
    if EMBEDDING_PROVIDER == "huggingface":
        return _embed_huggingface(text_input)
    return _embed_ollama(text_input)


def _embed_ollama(text_input):
    url = f"{OLLAMA_URL}/api/embeddings"
    payload = {"model": EMBEDDING_MODEL, "prompt": text_input}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["embedding"]


def _embed_huggingface(text_input):
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"
    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    response = requests.post(url, headers=headers, json={
        "inputs": text_input,
        "options": {"wait_for_model": True}
    })
    response.raise_for_status()
    result = response.json()
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], float):
            return result
        if isinstance(result[0], list):
            if isinstance(result[0][0], float):
                return result[0]
            n = len(result[0])
            dim = len(result[0][0])
            return [sum(result[0][t][d] for t in range(n)) / n for d in range(dim)]
    return result


model = Groq(model_id=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))


def get_knowledge_base():
    collections = load_collections()
    return MultiKnowledgeBase(
        db_url=DB_URL, collections=collections,
        embed_fn=get_embedding
    )


# ── Pages ──

@app.route('/')
def index():
    return render_template('index.html')


# ── Health ──

@app.route('/api/health')
def health():
    status = {"postgres": False, "embeddings": False}
    try:
        from sqlalchemy import create_engine, text as sql_text
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            conn.execute(sql_text("SELECT 1"))
        status["postgres"] = True
    except Exception:
        pass
    try:
        if EMBEDDING_PROVIDER == "huggingface":
            test = _embed_huggingface("test")
            status["embeddings"] = isinstance(test, list) and len(test) > 0
        else:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            status["embeddings"] = r.status_code == 200
    except Exception:
        pass
    status["ok"] = all([status["postgres"], status["embeddings"]])
    return jsonify(status)


# ── Chat ──

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        chat_history = data.get('history', [])

        kb = get_knowledge_base()
        context_chunks = kb.get_context(user_message, top_k=5)

        context = "\n\n".join(context_chunks) if context_chunks else ""
        prompt = f"Context:\n{context}\n\nUser: {user_message}\nAssistant:" if context else user_message

        response = model.complete(prompt, history=chat_history)

        return jsonify({'response': response, 'context_used': len(context_chunks) > 0})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Collections CRUD ──

@app.route('/api/collections', methods=['GET'])
def list_collections():
    collections = load_collections()
    result = []
    for name in collections:
        try:
            vdb = PgVector2(collection=name, db_url=DB_URL, embedding_dim=EMBEDDING_DIM)
            count = vdb.get_count()
            result.append({"name": name, "chunks": count})
        except Exception:
            result.append({"name": name, "chunks": 0, "error": True})
    return jsonify(result)


@app.route('/api/collections/<name>', methods=['DELETE'])
def delete_collection(name):
    if not validate_collection_name(name):
        return jsonify({"error": "Invalid collection name"}), 400
    try:
        vdb = PgVector2(collection=name, db_url=DB_URL, embedding_dim=EMBEDDING_DIM)
        vdb.drop_table()
        remove_collection(name)
        return jsonify({"status": "deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Ingestion: File Upload ──

@app.route('/api/collections/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    original_name = file.filename
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in ('.pdf', '.pptx'):
        return jsonify({"error": "Only PDF and PPTX files are supported"}), 400

    collection_name = request.form.get('collection_name', '').strip()
    if not collection_name:
        base = os.path.splitext(original_name)[0]
        collection_name = re.sub(r'[^a-zA-Z0-9]', '_', base).strip('_').lower()
        if not collection_name or not collection_name[0].isalpha():
            collection_name = 'doc_' + collection_name

    if not validate_collection_name(collection_name):
        return jsonify({"error": "Name must start with a letter, only letters/numbers/underscores"}), 400

    filename = secure_filename(file.filename) or f"{collection_name}{ext}"

    filepath = os.path.join(UPLOAD_FOLDER, f"{collection_name}_{filename}")
    file.save(filepath)

    job_id = str(uuid.uuid4())[:8]
    ingestion_jobs[job_id] = {
        "status": "queued", "collection": collection_name,
        "filename": filename, "type": "file"
    }

    thread = threading.Thread(target=_ingest_file, args=(filepath, collection_name, job_id), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued", "collection": collection_name})


# ── Ingestion: Web URL ──

@app.route('/api/collections/web', methods=['POST'])
def ingest_web():
    data = request.json
    url = data.get('url', '').strip()
    collection_name = data.get('collection_name', '').strip()

    if not url:
        return jsonify({"error": "URL is required"}), 400
    if not collection_name:
        return jsonify({"error": "Collection name is required"}), 400
    if not validate_collection_name(collection_name):
        return jsonify({"error": "Name must start with a letter, only letters/numbers/underscores"}), 400

    job_id = str(uuid.uuid4())[:8]
    ingestion_jobs[job_id] = {
        "status": "queued", "collection": collection_name,
        "url": url, "type": "web"
    }

    thread = threading.Thread(target=_ingest_web, args=(url, collection_name, job_id), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})


# ── Ingestion Status ──

@app.route('/api/ingestion/<job_id>')
def ingestion_status(job_id):
    job = ingestion_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route('/api/ingestion')
def all_ingestion_jobs():
    return jsonify(ingestion_jobs)


# ── Background Workers ──

def _ingest_file(filepath, collection_name, job_id):
    try:
        ingestion_jobs[job_id]["status"] = "reading"
        ext = os.path.splitext(filepath)[1].lower()

        vdb = PgVector2(collection=collection_name, db_url=DB_URL, embedding_dim=EMBEDDING_DIM)

        if ext == '.pptx':
            kb = PPTXKnowledgeBase(path=filepath, vector_db=vdb, reader=PPTXReader(chunk=True))
        else:
            kb = PDFKnowledgeBase(path=filepath, vector_db=vdb, reader=SimplePDFReader(chunk=True))

        ingestion_jobs[job_id]["status"] = "embedding"
        kb.load(embed_fn=get_embedding)
        add_collection(collection_name)
        ingestion_jobs[job_id]["status"] = "complete"
    except Exception as e:
        ingestion_jobs[job_id]["status"] = "error"
        ingestion_jobs[job_id]["error"] = str(e)


def _ingest_web(url, collection_name, job_id):
    try:
        ingestion_jobs[job_id]["status"] = "scraping"
        vdb = PgVector2(collection=collection_name, db_url=DB_URL, embedding_dim=EMBEDDING_DIM)
        kb = WebKnowledgeBase(url=url, vector_db=vdb, reader=WebReader(chunk=True))

        ingestion_jobs[job_id]["status"] = "embedding"
        kb.load(embed_fn=get_embedding)
        add_collection(collection_name)
        ingestion_jobs[job_id]["status"] = "complete"
    except Exception as e:
        ingestion_jobs[job_id]["status"] = "error"
        ingestion_jobs[job_id]["error"] = str(e)


if __name__ == '__main__':
    collections = load_collections()
    print("\n  Chat AXE - RAG Chatbot")
    print("  ----------------------")
    print(f"  Database:    {DB_URL}")
    print(f"  Embeddings:  {EMBEDDING_PROVIDER} ({EMBEDDING_MODEL}, {EMBEDDING_DIM}d)")
    print(f"  Groq key:    {'configured' if GROQ_API_KEY else 'MISSING - set GROQ_API_KEY in .env'}")
    print(f"  Collections: {collections if collections else '(none yet - upload from the UI)'}")
    print(f"\n  Open http://localhost:5000\n")
    app.run(debug=True, port=5000)
