# mini_rag.py - a complete, minimal RAG in one file.
import sys, re, requests, faiss
from sentence_transformers import SentenceTransformer

def load_text(path):
    if path.endswith(".pdf"):
        import pymupdf
        doc = pymupdf.open(path); text = "\n".join(p.get_text() for p in doc); doc.close(); return text
    return open(path, encoding="utf-8").read()

def chunk(text, size=500):
    chunks, cur = [], ""
    for s in re.split(r'(?<=[.!?])\s+', text):
        if len(cur) + len(s) < size: cur += " " + s
        else:
            if cur.strip(): chunks.append(cur.strip())
            cur = s
    if cur.strip(): chunks.append(cur.strip())
    return [c for c in chunks if len(c) > 40]

print("Loading embedding model (first run downloads it)...")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
def embed(texts, is_query=False):
    if is_query: texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
    return embedder.encode(texts, normalize_embeddings=True).astype("float32")

path = sys.argv[1] if len(sys.argv) > 1 else "sample.txt"
chunks = chunk(load_text(path))
print(f"Indexed {len(chunks)} chunks from {path}")
vecs = embed(chunks)
index = faiss.IndexFlatL2(vecs.shape[1]); index.add(vecs)

def ask(question, k=3):
    _, ids = index.search(embed([question], is_query=True), k)
    context = "\n---\n".join(chunks[i] for i in ids[0])
    prompt = f"Answer using ONLY this context. If it isn't there, say so.\n\nContext:\n{context}\n\nQuestion: {question}"
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": "llama3.1:8b", "prompt": prompt, "stream": False}, timeout=120)
    return r.json()["response"]

print("Ask a question (or 'quit'):")
while True:
    q = input("> ").strip()
    if q.lower() in ("quit", "exit", ""): break
    print("\n" + ask(q) + "\n")
