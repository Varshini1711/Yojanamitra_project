"""
ingest.py - Load schemes.json into ChromaDB with embeddings.
Run once before starting the backend server, or whenever schemes.json is updated.
Usage: python rag/ingest.py
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Model loaded!")

client = chromadb.PersistentClient(path="./chroma_db")

# Always start fresh
try:
    client.delete_collection(name="schemes")
    print("Old collection deleted.")
except:
    pass

collection = client.create_collection(name="schemes")

print("Loading schemes from JSON file...")
with open("data/schemes.json", "r", encoding="utf-8") as f:
    schemes = json.load(f)

print(f"Found {len(schemes)} schemes. Storing...")

for i, scheme in enumerate(schemes):

    # Handle eligibility — could be list or string
    eligibility = scheme.get("eligibility", "Not specified")
    if isinstance(eligibility, list):
        eligibility_text = "\n    - " + "\n    - ".join(eligibility)
    else:
        eligibility_text = str(eligibility)

    # Handle documents_required — could be list or string
    documents = scheme.get("documents_required", "Not specified")
    if isinstance(documents, list):
        documents_text = "\n    - " + "\n    - ".join(documents)
    else:
        documents_text = str(documents)

    # Handle application_process — correct field name
    how_to_apply = (
        scheme.get("application_process") or
        scheme.get("how_to_apply") or
        "Visit official website for application details"
    )

    # Build rich text for embedding
    text_to_embed = f"""
Scheme Name: {scheme.get('name', 'Unknown')}
Category: {scheme.get('category', 'Not specified')}
Ministry: {scheme.get('ministry', 'Not specified')}
Description: {scheme.get('description', 'Not specified')}

Eligibility: {eligibility_text}

Benefits: {scheme.get('benefits', 'Not specified')}

Documents Required: {documents_text}

How to Apply: {how_to_apply}

Official Website: {scheme.get('source_url', 'https://myscheme.gov.in')}
State: {scheme.get('state', 'All India')}
""".strip()

    embedding = model.encode(text_to_embed).tolist()

    collection.add(
        documents=[text_to_embed],
        embeddings=[embedding],
        metadatas=[{
            "name": scheme.get("name", "Unknown"),
            "category": scheme.get("category", "Not specified"),
            "state": scheme.get("state", "All India"),
            "source_url": scheme.get("source_url", "https://myscheme.gov.in"),
            "last_verified": (
                scheme.get("last_verified_date") or
                scheme.get("last_verified") or
                "2024-12-01"
            ),
            "ministry": scheme.get("ministry", "Not specified"),
            "description": scheme.get("description", "")[:200],
        }],
        ids=[f"scheme_{i}"]
    )
    print(f"  ✅ Stored: {scheme.get('name')}")

print(f"\n✅ Done! {collection.count()} schemes stored in ChromaDB.")
print("📁 'chroma_db' folder updated.")