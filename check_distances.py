import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="schemes")

queries = [
    "MUDRA loan for small business",
    "scholarship for SC students",
    "PM Kisan farmer scheme",
    "Yuva Nidhi unemployment Karnataka",
    "SSP Karnataka scholarship portal"
]

for q in queries:
    emb = model.encode(q).tolist()
    r = collection.query(query_embeddings=[emb], n_results=1, include=["metadatas", "distances"])
    print(f"{q[:40]} -> {r['metadatas'][0][0]['name'][:30]} | Distance: {round(r['distances'][0][0], 2)}")