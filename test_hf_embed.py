import os
from dotenv import load_dotenv
load_dotenv()
import requests

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
MODEL = "BAAI/bge-small-en-v1.5"

url = f"https://router.huggingface.co/hf-inference/models/{MODEL}"
headers = {"Content-Type": "application/json"}
if HF_API_TOKEN:
    headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

print(f"URL: {url}")
print(f"Token present: {bool(HF_API_TOKEN)}")

try:
    resp = requests.post(url, headers=headers, json={
        "inputs": "test query",
        "options": {"wait_for_model": True}
    }, timeout=30)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text[:500]}")
    if resp.status_code == 200:
        result = resp.json()
        if isinstance(result, list):
            if isinstance(result[0], float):
                print(f"SUCCESS - dim: {len(result)}")
            elif isinstance(result[0], list):
                if isinstance(result[0][0], float):
                    print(f"SUCCESS - dim: {len(result[0])}")
                else:
                    print(f"Token-level: {len(result[0])}x{len(result[0][0])}")
except Exception as e:
    print(f"ERROR: {e}")
