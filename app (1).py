"""
Query Lambda — 100% FREE, no API keys needed
Uses sentence-transformers for embeddings + Ollama (local LLM) for answers
Triggered by API Gateway POST /ask
"""

import os
import json
import logging
import requests
from sentence_transformers import SentenceTransformer
import chromadb

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH       = os.environ.get("CHROMA_PATH", "/mnt/efs/chroma")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "rag_docs")
EMBED_MODEL_NAME  = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
OLLAMA_URL        = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL", "mistral")
TOP_K             = int(os.environ.get("TOP_K", "3"))

# ── Load embedding model once (reused on warm Lambda) ────────────────────────
logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
logger.info("Embedding model loaded")

# ── CORS headers ──────────────────────────────────────────────────────────────
HEADERS = {
    "Content-Type":                "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}

# ── RAG prompt ────────────────────────────────────────────────────────────────
RAG_PROMPT = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."
Do not make things up.

Context:
{context}

Question: {question}

Answer:"""


def retrieve_context(question: str):
    """Embed question and retrieve top-k matching chunks from ChromaDB."""
    query_vector = embed_model.encode([question])[0].tolist()

    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=CHROMA_COLLECTION)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    logger.info(f"Retrieved {len(documents)} chunks. Best distance: {distances[0]:.4f}")
    return documents, metadatas


def generate_answer_ollama(question: str, context_chunks: list) -> str:
    """
    Call Ollama (local LLM — Mistral 7B) for FREE answer generation.
    Ollama must be running on the same machine or accessible via OLLAMA_URL.
    """
    context = "\n\n---\n\n".join(context_chunks)
    prompt  = RAG_PROMPT.format(context=context, question=question)

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,      # deterministic answers
                    "num_predict": 512,    # max tokens in reply
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("response", "No response from model.")

    except requests.exceptions.ConnectionError:
        # Fallback if Ollama is not running — return context directly
        logger.warning("Ollama not reachable — returning raw context as answer")
        return (
            "⚠️ Local LLM (Ollama) is not running. "
            "Start it with: ollama serve\n\n"
            "Relevant context found:\n\n" + context_chunks[0]
        )


def handler(event, context):
    """
    Lambda entry point.
    POST /ask  body: {"question": "..."}
    """
    # Handle CORS preflight
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": HEADERS, "body": ""}

    try:
        body     = json.loads(event.get("body") or "{}")
        question = body.get("question", "").strip()

        if not question:
            return {
                "statusCode": 400,
                "headers":    HEADERS,
                "body":       json.dumps({"error": "Missing 'question' in request body"}),
            }

        logger.info(f"Question: {question}")

        # Retrieve relevant chunks from ChromaDB
        chunks, metadatas = retrieve_context(question)

        # Generate answer using free local Ollama LLM
        answer = generate_answer_ollama(question, chunks)

        # Build source list
        sources = [
            {
                "text":   chunk[:200] + "..." if len(chunk) > 200 else chunk,
                "source": meta.get("source", "unknown"),
                "page":   meta.get("page", 0),
            }
            for chunk, meta in zip(chunks, metadatas)
        ]

        return {
            "statusCode": 200,
            "headers":    HEADERS,
            "body":       json.dumps({
                "answer":   answer,
                "sources":  sources,
                "question": question,
                "model":    OLLAMA_MODEL,
            }),
        }

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "headers":    HEADERS,
            "body":       json.dumps({"error": str(e)}),
        }
