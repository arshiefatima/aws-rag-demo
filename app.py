"""
Ingestion Lambda — 100% FREE, no API keys needed
Uses sentence-transformers for local embeddings (no OpenAI)
Triggered by S3 upload → chunks → embeds → stores in ChromaDB on EFS
"""

import os
import json
import logging
import boto3
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Config (env vars — no API keys!) ─────────────────────────────────────────
CHROMA_PATH        = os.environ.get("CHROMA_PATH", "/mnt/efs/chroma")
CHROMA_COLLECTION  = os.environ.get("CHROMA_COLLECTION", "rag_docs")
EMBED_MODEL_NAME   = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE         = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP      = int(os.environ.get("CHUNK_OVERLAP", "50"))

# ── AWS client ────────────────────────────────────────────────────────────────
s3 = boto3.client("s3")

# ── Load embedding model once (reused across warm Lambda invocations) ─────────
# all-MiniLM-L6-v2 is tiny (80MB), fast, and produces great embeddings for FREE
logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
logger.info("Embedding model loaded successfully")


def load_document(local_path: str):
    """Load PDF or TXT file into LangChain documents."""
    if local_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(local_path)
    else:
        loader = TextLoader(local_path, encoding="utf-8")
    return loader.load()


def chunk_documents(docs):
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def embed_and_store(chunks, file_key: str) -> int:
    """
    Embed chunks using FREE local sentence-transformers model
    and persist to ChromaDB on EFS.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [chunk.page_content for chunk in chunks]

    # Batch embed — sentence-transformers handles batching internally
    logger.info(f"Embedding {len(texts)} chunks with {EMBED_MODEL_NAME}...")
    vectors = embed_model.encode(texts, show_progress_bar=False).tolist()

    ids       = [f"{file_key}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source":      file_key,
            "chunk_index": i,
            "page":        chunk.metadata.get("page", 0),
        }
        for i, chunk in enumerate(chunks)
    ]

    # Upsert — safe to re-upload same file
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=vectors,
        metadatas=metadatas,
    )

    logger.info(f"Stored {len(chunks)} chunks from '{file_key}'")
    return len(chunks)


def handler(event, context):
    """
    Lambda entry point — triggered by S3 PutObject event.
    """
    try:
        record = event["Records"][0]
        bucket = record["s3"]["bucket"]["name"]
        key    = record["s3"]["object"]["key"]
        logger.info(f"Processing s3://{bucket}/{key}")

        # Only handle PDF and TXT
        if not key.lower().endswith((".pdf", ".txt")):
            logger.warning(f"Skipping unsupported file: {key}")
            return {"statusCode": 200, "body": "Skipped — unsupported file type"}

        # Download to Lambda /tmp
        filename   = key.split("/")[-1]
        local_path = f"/tmp/{filename}"
        s3.download_file(bucket, key, local_path)
        logger.info(f"Downloaded to {local_path}")

        # Pipeline: load → chunk → embed → store
        docs   = load_document(local_path)
        chunks = chunk_documents(docs)
        count  = embed_and_store(chunks, key)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message":       "Ingestion successful",
                "file":          key,
                "chunks_stored": count,
                "embed_model":   EMBED_MODEL_NAME,
            }),
        }

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }
