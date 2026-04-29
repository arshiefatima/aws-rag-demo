# RAG Demo — 100% Free, No API Keys Needed

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Free](https://img.shields.io/badge/Cost-100%25%20Free-brightgreen)
![LangChain](https://img.shields.io/badge/LangChain-0.1-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-purple)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20Mistral-orange)
![AWS](https://img.shields.io/badge/AWS-Lambda%20%2B%20S3-orange)

A production-grade **Retrieval-Augmented Generation (RAG)** system — runs completely free on your laptop OR deploys to AWS. No OpenAI key, no paid APIs, zero cost.

---

## Free Tech Stack

| Component | Free Tool | Why |
|-----------|-----------|-----|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | 80MB, runs locally, great quality |
| LLM | Ollama + Mistral 7B | Runs on your CPU/GPU, 100% private |
| Vector DB | ChromaDB | Embedded, no server needed |
| Orchestration | LangChain | Open source |
| Cloud (optional) | AWS Lambda + S3 + EFS | Free tier covers light usage |

---

## Architecture

```
Your Documents (PDF / TXT)
         │
         ▼
  [Local] run_local.py ingest    OR    [AWS] S3 Upload → Lambda
         │                                      │
   sentence-transformers                sentence-transformers
   (free local embeddings)              (free local embeddings)
         │                                      │
         ▼                                      ▼
    ChromaDB (local)                   ChromaDB (AWS EFS)
         │                                      │
         ▼                                      ▼
  [Local] run_local.py ask       OR    [AWS] API Gateway → Lambda
         │                                      │
    Ollama Mistral 7B                  Ollama Mistral 7B
    (free local LLM)                   (or swap in any LLM)
         │                                      │
         ▼                                      ▼
       Answer ✓                             Answer ✓
```

---

## Quick Start — Local (No AWS Needed)

### Step 1: Install dependencies
```bash
git clone https://github.com/arshiefatima/aws-rag-demo
cd aws-rag-demo
pip install -r ingest/requirements.txt -r query/requirements.txt
```

### Step 2: Install Ollama (free local LLM)
```bash
# Mac
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows — download from https://ollama.com/download
```

### Step 3: Pull Mistral model and start Ollama
```bash
ollama pull mistral       # downloads ~4GB once
ollama serve              # keep this running in a separate terminal
```

### Step 4: Ingest your documents
```bash
# Put any PDF or TXT in the docs/ folder, then:
python run_local.py ingest docs/your_resume.pdf
```

### Step 5: Ask questions
```bash
python run_local.py ask "What are the main skills in this resume?"

# Or interactive chat mode:
python run_local.py chat
```

---

## Example Output

```
Question: What programming languages does Arshie know?

Answer:
Based on the resume, Arshie knows Python, Core Java, C, and C++.
She also has experience with JavaScript, HTML, and CSS for web technologies.

Sources used:
  [1] resume.pdf (page 1) — Programming Languages: Python, Core Java, C, C++...
  [2] resume.pdf (page 1) — Web Technologies: JavaScript, HTML, JSON, CSS...
```

---

## Deploy to AWS (Optional)

If you want a live URL to share:
```bash
sam build
sam deploy --guided
# Follow prompts — no API key parameters needed!
# Copy the ApiUrl output and add to .env as API_URL
```

---

## Project Structure

```
aws-rag-demo/
├── run_local.py           ← START HERE — full local pipeline
├── ingest/
│   ├── app.py             ← Lambda: S3 → chunk → embed → ChromaDB
│   └── requirements.txt
├── query/
│   ├── app.py             ← Lambda: question → retrieve → Ollama → answer
│   └── requirements.txt
├── tests/
│   ├── test_ingest.py
│   └── test_query.py
├── scripts/
│   ├── upload_docs.sh
│   └── test_api.sh
├── docs/                  ← Put your PDFs/TXTs here
├── template.yaml          ← AWS SAM (optional cloud deploy)
├── Makefile
└── .env.example
```

---

## Author

**Arshie Fatima** — AI Engineer  
[GitHub](https://github.com/arshiefatima) · [LinkedIn](https://www.linkedin.com/in/arshie-fatima-1707361ab)
