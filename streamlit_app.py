"""
Streamlit Web UI for the RAG Demo
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import os
import tempfile

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🤖",
    layout="centered"
)

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH       = "./chroma_local"
CHROMA_COLLECTION = "rag_docs"
EMBED_MODEL_NAME  = "all-MiniLM-L6-v2"
OLLAMA_URL        = "http://localhost:11434"
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 50
TOP_K             = 3

RAG_PROMPT = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information about that."

Context:
{context}

Question: {question}

Answer:"""

# ── Load models (cached so they only load once) ───────────────────────────────
@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

# ── Get available Ollama models ───────────────────────────────────────────────
def get_ollama_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except:
        return []

# ── Ingest a file ─────────────────────────────────────────────────────────────
def ingest_file(uploaded_file, embed_model, collection):
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix == ".pdf":
        docs = PyPDFLoader(tmp_path).load()
    else:
        docs = TextLoader(tmp_path, encoding="utf-8").load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    texts  = [c.page_content for c in chunks]
    vectors = embed_model.encode(texts, show_progress_bar=False).tolist()
    ids     = [f"{uploaded_file.name}_chunk_{i}" for i in range(len(chunks))]
    metas   = [{"source": uploaded_file.name, "page": c.metadata.get("page", 0)} for c in chunks]

    collection.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)
    os.unlink(tmp_path)
    return len(chunks)

# ── Ask a question ────────────────────────────────────────────────────────────
def ask_question(question, model_name, embed_model, collection):
    query_vector = embed_model.encode([question])[0].tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not chunks:
        return "No documents found. Please upload a document first.", []

    context = "\n\n---\n\n".join(chunks)
    prompt  = RAG_PROMPT.format(context=context, question=question)

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 512},
            },
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        answer = "Ollama is not running. Start it with: ollama serve"
    except Exception as e:
        answer = f"Error: {str(e)}"

    sources = [
        {"text": c[:150] + "..." if len(c) > 150 else c,
         "source": m.get("source", "?"), "page": m.get("page", 0)}
        for c, m in zip(chunks, metadatas)
    ]
    return answer, sources

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🤖 RAG Document Q&A")
st.caption("Upload a document, ask questions — powered by local AI, no API key needed")

# Load models
embed_model = load_embed_model()
collection  = load_chroma()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    models = get_ollama_models()
    if models:
        selected_model = st.selectbox("Ollama Model", models, index=0)
        st.success(f"Ollama connected")
    else:
        selected_model = st.text_input("Model name", value="mistral:latest")
        st.warning("Ollama not detected. Make sure it's running.")

    st.divider()
    st.header("📄 Upload Document")
    uploaded = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        help="Your document will be chunked, embedded and stored locally"
    )

    if uploaded:
        if st.button("📥 Ingest Document", use_container_width=True):
            with st.spinner(f"Processing {uploaded.name}..."):
                count = ingest_file(uploaded, embed_model, collection)
            st.success(f"Stored {count} chunks from {uploaded.name}")

    st.divider()
    count_info = collection.count()
    st.metric("Chunks in database", count_info)
    if st.button("🗑️ Clear database", use_container_width=True):
        col = chromadb.PersistentClient(path=CHROMA_PATH)
        col.delete_collection(CHROMA_COLLECTION)
        st.cache_resource.clear()
        st.rerun()

# Main chat area
st.subheader("💬 Ask a question")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources used"):
                for s in msg["sources"]:
                    st.caption(f"**{s['source']}** (page {s['page']}) — {s['text']}")

# Chat input
if prompt := st.chat_input("Ask anything about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = ask_question(prompt, selected_model, embed_model, collection)
        st.write(answer)
        if sources:
            with st.expander("📚 Sources used"):
                for s in sources:
                    st.caption(f"**{s['source']}** (page {s['page']}) — {s['text']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

# Quick question buttons
if collection.count() > 0:
    st.divider()
    st.caption("Quick questions:")
    cols = st.columns(3)
    quick_qs = [
        "What are the top skills?",
        "What companies did they work at?",
        "What projects were built?",
    ]
    for i, q in enumerate(quick_qs):
        if cols[i].button(q, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            answer, sources = ask_question(q, selected_model, embed_model, collection)
            st.session_state.messages.append({
                "role": "assistant", "content": answer, "sources": sources
            })
            st.rerun()
