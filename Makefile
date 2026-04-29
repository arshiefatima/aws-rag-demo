.PHONY: install setup test run-ingest ask chat build deploy clean help

# Install all dependencies
install:
	pip install -r ingest/requirements.txt
	pip install -r query/requirements.txt
	pip install pytest

# Setup Ollama + pull Mistral model (run once)
setup:
	@echo "Installing Ollama..."
	@echo "Mac:   brew install ollama"
	@echo "Linux: curl -fsSL https://ollama.ai/install.sh | sh"
	@echo ""
	@echo "After installing, run:"
	@echo "  ollama pull mistral"
	@echo "  ollama serve        (keep this running in a separate terminal)"

# Run unit tests
test:
	pytest tests/ -v --tb=short

# Ingest a document (usage: make ingest FILE=docs/resume.pdf)
ingest:
	python run_local.py ingest $(FILE)

# Ask a question (usage: make ask Q="What are the skills?")
ask:
	python run_local.py ask "$(Q)"

# Interactive chat mode
chat:
	python run_local.py chat

# Upload docs to S3 (for cloud deployment)
upload:
	./scripts/upload_docs.sh

# Test deployed API
test-api:
	./scripts/test_api.sh "$(Q)"

# Build SAM package for AWS deployment
build:
	sam build

# Deploy to AWS
deploy:
	sam deploy --guided

# Clean up
clean:
	rm -rf .aws-sam/ chroma_local/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

help:
	@echo "Commands:"
	@echo "  make install              Install dependencies"
	@echo "  make setup                Setup Ollama instructions"
	@echo "  make test                 Run unit tests"
	@echo "  make ingest FILE=<path>   Ingest a document"
	@echo "  make ask Q='<question>'   Ask a question"
	@echo "  make chat                 Interactive Q&A mode"
	@echo "  make deploy               Deploy to AWS"
	@echo "  make clean                Remove build artifacts"
