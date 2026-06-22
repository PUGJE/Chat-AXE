from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from sqlalchemy import create_engine, text
import numpy as np
import requests
import os
db_url = os.getenv("DATABASE_URL", "postgresql+psycopg2://cu:cu@localhost:5532/cu")
engine = create_engine(db_url)

with engine.connect() as conn:
    result = conn.execute(text("SELECT 1;"))
    print(result.fetchone())


def build_rag_prompt(context_chunks, user_query):
    context = "\n\n".join(context_chunks)
    prompt = (
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\n"
        f"Answer:"
    )
    return prompt


def get_llm_answer(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",  # or whichever model you want
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["response"]

def get_top_k_chunks(query_embedding, k=5):
    # Convert embedding to PostgreSQL vector format
    embedding_str = ','.join(map(str, query_embedding))
    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT content, embedding <#> :embedding AS distance
                FROM documents
                ORDER BY distance ASC
                LIMIT :k
            """),
            {"embedding": f'[{embedding_str}]', "k": k}
        )
        return [row[0] for row in result]

def get_embedding(text):
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": "nomic-embed-text",
        "prompt": text
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["embedding"]

def insert_chunk(content, embedding):
    embedding_str = ','.join(map(str, embedding))
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO documents (content, embedding) VALUES (:content, :embedding) ON CONFLICT (content) DO NOTHING"),
            {"content": content, "embedding": f'[{embedding_str}]'}
        )

def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()

def load_pptx(path):
    loader = UnstructuredPowerPointLoader(path)
    return loader.load()

def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def ingest_documents():
    file_path = input("Enter the path to the PDF or PPTX file to ingest: ").strip()
    print(f"Received file path: '{file_path}'")
    if file_path.lower().endswith('.pdf'):
        docs = load_pdf(file_path)
    elif file_path.lower().endswith('.pptx'):
        docs = load_pptx(file_path)
    else:
        print("Unsupported file type. Only PDF and PPTX are supported.")
        return
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} documents, chunked into {len(chunks)} pieces.")
    print("First chunk:", chunks[0].page_content[:300])
    for chunk in chunks:
        chunk_text = chunk.page_content
        embedding = get_embedding(chunk_text)
        insert_chunk(chunk_text, embedding)
        print("Inserted chunk.")
    print("All chunks inserted into the database.")

def run_chatbot():
    while True:
        user_query = input("Ask a question (or type 'exit'): ")
        if user_query.lower() == "exit":
            break
        query_embedding = get_embedding(user_query)
        top_chunks = get_top_k_chunks(query_embedding, k=5)
        prompt = build_rag_prompt(top_chunks, user_query)
        answer = get_llm_answer(prompt)
        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    mode = input("Type 'ingest' to add documents, or 'chat' to ask questions: ").strip().lower()
    if mode == "ingest":
        ingest_documents()
    elif mode == "chat":
        run_chatbot()
    else:
        print("Unknown mode. Please type 'ingest' or 'chat'.")
