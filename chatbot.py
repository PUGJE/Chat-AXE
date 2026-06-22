import os
import openai
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from types import SimpleNamespace
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

from knowledgebase import (
    OCRPDFReader, WebReader, PDFKnowledgeBase, WebKnowledgeBase, MultiKnowledgeBase, PgVector2,
    load_collections, add_collection, PPTXReader, PPTXKnowledgeBase
)


class Groq:
    def __init__(self, model_id, api_key=None):
        self.model_id = model_id
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def complete(self, prompt, history=None):
        messages = [
            {"role": "system", "content": "You are a Chandigarh University Assistant. Your role is to help students and faculty with academic inquiries, course information, and administrative tasks. Provide clear, concise answers based on available resources."},
            *([{"role": h["role"], "content": h["content"]} for h in (history or [])]),
            {"role": "user", "content": prompt}
        ]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_id,
            "messages": messages
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class PgAssistantStorage:
    def __init__(self, table_name, db_url):
        self.table_name = table_name
        self.db_url = db_url
    def get_all_run_ids(self, user):
        return []


class Agent:
    def __init__(self, model):
        self.model = model

    def complete(self, prompt, history=None):
        # history is a list of {"role": "user"/"assistant", "content": "..."}
        return self.model.complete(prompt, history=history)

class Assistant:
    def __init__(
        self,
        run_id,
        user_id,
        knowledge_base,
        storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
    ):
        self.run_id = run_id
        self.user_id = user_id
        self.knowledge_base = knowledge_base
        self.storage = storage
        self.show_tool_calls = show_tool_calls
        self.search_knowledge = search_knowledge
        self.read_chat_history = read_chat_history
        self.history = []

    def cli_app(self, markdown=True):
        print("Assistant ready! Type your questions. Type 'exit' to quit.")
        while True:
            user_query = input("You: ")
            if user_query.lower() == "exit":
                break

            # Optionally search knowledge base
            context_chunks = []
            if self.search_knowledge:
                if isinstance(self.knowledge_base, MultiKnowledgeBase):
                    context_chunks = self.knowledge_base.get_context(user_query)
                else:
                    # Fallback for single knowledge base
                    query_embedding = self.knowledge_base.vector_db.query_embedding(user_query)
                    context_chunks = self.knowledge_base.vector_db.query(query_embedding, top_k=5)

            # Build prompt with context and history
            prompt = self.build_prompt(context_chunks, user_query)
            answer = self.knowledge_base.agent.complete(prompt, history=self.history if self.read_chat_history else None)
            print("\nAssistant:", answer, "\n")

            # Save to history
            self.history.append({"role": "user", "content": user_query})
            self.history.append({"role": "assistant", "content": answer})

    def build_prompt(self, context_chunks, user_query):
        context = "\n\n".join(context_chunks) if context_chunks else ""
        history_text = ""
        for turn in self.history:
            history_text += f"{turn['role'].capitalize()}: {turn['content']}\n"
        prompt = (
            f"Context:\n{context}\n\n"
            f"Conversation so far:\n{history_text}\n"
            f"User: {user_query}\n"
            f"Assistant:"
        )
        return prompt


import requests
from bs4 import BeautifulSoup
import re

def get_embedding(text):
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": "nomic-embed-text:latest",
        "prompt": text
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["embedding"]


if __name__ == "__main__":
    db_url = "postgresql+psycopg2://cu:cu@127.0.0.1:5532/cu"
    model = Groq(model_id="llama3-70b-8192")
    agent = Agent(model=model)

    # Mode selection
    mode = input("Type 'ingest' to process a new PDF, 'web' to ingest a web page, or 'chat' to start the chatbot: ").strip().lower()
    
    if mode == "ingest":
        # Process and embed a new PDF or PPTX
        file_path = input("Enter the path to the PDF or PPTX file: ").strip()
        collection_name = input("Enter collection name for the database: ").strip()
        
        if file_path.lower().endswith('.pptx'):
            knowledge_base = PPTXKnowledgeBase(
                path=file_path,
                vector_db=PgVector2(collection=collection_name, db_url=db_url),
                reader=PPTXReader(chunk=True)
            )
        else:
            knowledge_base = PDFKnowledgeBase(
                path=file_path,
                vector_db=PgVector2(collection=collection_name, db_url=db_url),
                reader=OCRPDFReader(chunk=True)
            )
        print("Loading and embedding file...")
        knowledge_base.load(embed_fn=get_embedding)
        print("File content loaded into vector DB.")
        add_collection(collection_name)
        knowledge_base.agent = agent
        
    elif mode == "web":
        # Ingest a web page
        url = input("Enter the URL of the web page to ingest: ").strip()
        collection_name = input("Enter collection name for the database: ").strip()
        
        knowledge_base = WebKnowledgeBase(
            url=url,
            vector_db=PgVector2(collection=collection_name, db_url=db_url),
            reader=WebReader(chunk=True)
        )
        print("Loading and embedding web page...")
        knowledge_base.load(embed_fn=get_embedding)
        print("Web page content loaded into vector DB.")
        add_collection(collection_name)
        knowledge_base.agent = agent
        
    elif mode == "chat":
        # Start chatbot with automatic knowledge base selection
        print("Starting chatbot with automatic knowledge base selection...")
        
        # Load available collections from file
        available_collections = load_collections()
        if not available_collections:
            print("No collections found. Please ingest a PDF or web page first.")
            exit()
        
        # Create multi-knowledge base
        knowledge_base = MultiKnowledgeBase(db_url=db_url, collections=available_collections)
        knowledge_base.add_agent(agent)
        
    else:
        print("Invalid mode. Please type 'ingest', 'web', or 'chat'.")
        exit()
    
    storage = PgAssistantStorage(table_name=(available_collections[0] if mode == "chat" else collection_name), db_url=db_url)

    assistant = Assistant(
        run_id=None,
        user_id="user",
        knowledge_base=knowledge_base,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
    )
    assistant.cli_app(markdown=True)
