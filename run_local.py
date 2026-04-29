"""
run_local.py — Run the ENTIRE RAG pipeline locally on your laptop
No AWS, no API keys, no cost!

Usage:
  python run_local.py ingest "docs/AI resume.pdf"
  python run_local.py ask "What are the main skills in the document?"
  python run_local.py chat
"""

import sys
import os
import json
import logging
import requests
from pathlib import Path

# ── Imports (compatible with latest LangChain) ────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH       = "./chroma_local"
CHROMA_COLLECTION = "rag_docs"
EMBED_MODEL_NAME  = "all-MiniLM-L6-v2"
OLLAMA_URL        = "http://localhost:11434"
OLLAMA_MODEL      = "mistral:latest"
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 50
TOP_K             = 3

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# ── Load embedding model ──────────────────────────────────────────────────────
print(f"Loading embedding model ({EMBED_MODEL_NAME})...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
print("Model ready.\n")

client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)

RAG_PROMPT = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""


# ── Ingest ────────────────────────────────────────────────────────────────────
def ingest(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"ERROR: File not found: {file_path}")
        print(f"Make sure the file is inside the docs/ folder.")
        sys.exit(1)

    print(f"Loading: {path.name}")
    if path.suffix.lower() == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    else:
        docs = TextLoader(str(path), encoding="utf-8").load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    texts   = [c.page_content for c in chunks]
    print("Embedding chunks (this takes ~10 seconds)...")
    vectors = embed_model.encode(texts, show_progress_bar=True).tolist()
    ids     = [f"{path.name}_chunk_{i}" for i in range(len(chunks))]
    metas   = [{"source": path.name, "page": c.metadata.get("page", 0)} for c in chunks]

    collection.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)
    print(f"\nDone! {len(chunks)} chunks stored in ChromaDB.")
    print("You can now run: python run_local.py ask \"your question here\"")


# ── Ask ───────────────────────────────────────────────────────────────────────
def ask(question: str):
    print(f"\nQuestion: {question}")
    print("-" * 60)

    query_vector = embed_model.encode([question])[0].tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not chunks:
        print("No documents found. Run: python run_local.py ingest <file>")
        return

    context = "\n\n---\n\n".join(chunks)
    prompt  = RAG_PROMPT.format(context=context, question=question)

    print(f"Generating answer with {OLLAMA_MODEL}...")
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 512},
            },
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        print("\nOllama is not running!")
        print("Open a NEW terminal window and run: ollama serve")
        print("\nShowing raw context instead:\n")
        answer = chunks[0]

    print(f"\nAnswer:\n{answer}")
    print("\n" + "-" * 60)
    print("Sources used:")
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas), 1):
        print(f"  [{i}] {meta.get('source')} (page {meta.get('page','?')}) — {chunk[:80]}...")


# ── Chat ──────────────────────────────────────────────────────────────────────
def chat():
    print("RAG Chat Mode — type 'exit' to quit\n")
    print(f"Model: {OLLAMA_MODEL} | Collection: {CHROMA_COLLECTION}")
    print("-" * 60)
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if question.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break
        if question:
            ask(question)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "ingest":
        if len(sys.argv) < 3:
            print('Usage: python run_local.py ingest "docs/your_file.pdf"')
            sys.exit(1)
        ingest(" ".join(sys.argv[2:]))

    elif command == "ask":
        if len(sys.argv) < 3:
            print('Usage: python run_local.py ask "your question here"')
            sys.exit(1)
        ask(" ".join(sys.argv[2:]))

    elif command == "chat":
        chat()

    else:
        print(f"Unknown command: {command}")
        print("Commands: ingest | ask | chat")
        sys.exit(1)
