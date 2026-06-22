import requests
import os
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
payload = {
    "model": "llama3",  # or "phi3"
    "prompt": "Hello, how are you?",
    "stream": False
}

response = requests.post(OLLAMA_URL, json=payload)
print(response.json())