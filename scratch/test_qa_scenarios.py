import requests
import json
import sys

# Ensure UTF-8 output encoding for Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://127.0.0.1:8000/api/v1"

print("Starting test...", flush=True)

try:
    print("\n================ TEST 1: Document Question with allow_web_search=True ================", flush=True)
    payload1 = {
        "question": "What tablets or medicines are prescribed in this prescription?",
        "document_id": None,
        "allow_web_search": True,
        "top_k": 4
    }
    r1 = requests.post(f"{BASE_URL}/search/ask", json=payload1, timeout=30)
    print(f"Status: {r1.status_code}", flush=True)
    data1 = r1.json()
    print(f"Source Type: {data1.get('source_type')}", flush=True)
    print(f"Answer:\n{data1.get('answer')}", flush=True)
    print(f"Doc Citations count: {len(data1.get('document_citations', []))}", flush=True)
    print(f"Web Citations count: {len(data1.get('web_citations', []))}", flush=True)

    print("\n================ TEST 2: General Question with allow_web_search=False ================", flush=True)
    payload2 = {
        "question": "Who is the Chief Minister of Tamil Nadu?",
        "document_id": None,
        "allow_web_search": False,
        "top_k": 4
    }
    r2 = requests.post(f"{BASE_URL}/search/ask", json=payload2, timeout=30)
    print(f"Status: {r2.status_code}", flush=True)
    data2 = r2.json()
    print(f"Source Type: {data2.get('source_type')}", flush=True)
    print(f"Answer:\n{data2.get('answer')}", flush=True)
    print(f"Doc Citations count: {len(data2.get('document_citations', []))}", flush=True)
    print(f"Web Citations count: {len(data2.get('web_citations', []))}", flush=True)

    print("\n================ TEST 3: General Question with allow_web_search=True ================", flush=True)
    payload3 = {
        "question": "Who is the Chief Minister of Tamil Nadu?",
        "document_id": None,
        "allow_web_search": True,
        "top_k": 4
    }
    r3 = requests.post(f"{BASE_URL}/search/ask", json=payload3, timeout=30)
    print(f"Status: {r3.status_code}", flush=True)
    data3 = r3.json()
    print(f"Source Type: {data3.get('source_type')}", flush=True)
    print(f"Answer:\n{data3.get('answer')}", flush=True)
    print(f"Doc Citations count: {len(data3.get('document_citations', []))}", flush=True)
    print(f"Web Citations count: {len(data3.get('web_citations', []))}", flush=True)

except Exception as e:
    print(f"Error during execution: {e}", flush=True)
