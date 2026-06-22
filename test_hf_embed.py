import os
from dotenv import load_dotenv
load_dotenv()

import requests

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"
headers = {"Content-Type": "application/json"}
if HF_API_TOKEN:
    headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

print(f"Calling: {url}")
print(f"Token present: {bool(HF_API_TOKEN)}")

try:
    response = requests.post(url, headers=headers, json={
        "inputs": "test",
        "options": {"wait_for_model": True}
    }, timeout=30)

    print(f"Status: {response.status_code}")
    print(f"Response (first 500 chars): {response.text[:500]}")

    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], float):
                print(f"SUCCESS - Embedding dim: {len(result)}")
            elif isinstance(result[0], list):
                if isinstance(result[0][0], float):
                    print(f"SUCCESS - Embedding dim: {len(result[0])}")
                elif isinstance(result[0][0], list):
                    print(f"Token-level embeddings - tokens: {len(result[0])}, dim: {len(result[0][0])}")
            else:
                print(f"Unexpected inner type: {type(result[0])}")
        else:
            print(f"Unexpected format: {type(result)}")
    else:
        print(f"FAILED with status {response.status_code}")
except Exception as e:
    print(f"ERROR: {e}")
