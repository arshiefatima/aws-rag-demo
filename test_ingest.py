"""
Unit tests for ingestion — run with: pytest tests/ -v
"""
import json
import pytest
from unittest.mock import patch, MagicMock, ANY


def make_s3_event(bucket, key):
    return {"Records": [{"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}]}


@patch.dict("os.environ", {"CHROMA_PATH": "/tmp/test_chroma", "CHROMA_COLLECTION": "test"})
@patch("ingest.app.embed_model")
@patch("ingest.app.chromadb.PersistentClient")
@patch("ingest.app.s3")
@patch("ingest.app.load_document")
@patch("ingest.app.chunk_documents")
def test_handler_pdf_success(mock_chunk, mock_load, mock_s3, mock_chroma, mock_embed):
    from ingest.app import handler

    mock_load.return_value  = [MagicMock(page_content="Resume content")]
    mock_chunk.return_value = [MagicMock(page_content="Chunk 1", metadata={"page": 1})]
    mock_embed.encode.return_value = [[0.1, 0.2, 0.3]]

    mock_col = MagicMock()
    mock_chroma.return_value.get_or_create_collection.return_value = mock_col

    response = handler(make_s3_event("my-bucket", "resume.pdf"), {})
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["chunks_stored"] == 1


@patch.dict("os.environ", {"CHROMA_PATH": "/tmp/test_chroma"})
def test_handler_skips_unsupported():
    from ingest.app import handler
    response = handler(make_s3_event("my-bucket", "photo.jpg"), {})
    assert response["statusCode"] == 200
    assert "Skipped" in response["body"]


@patch.dict("os.environ", {"CHROMA_PATH": "/tmp/test_chroma"})
@patch("ingest.app.s3")
def test_handler_returns_500_on_error(mock_s3):
    from ingest.app import handler
    mock_s3.download_file.side_effect = Exception("S3 error")
    response = handler(make_s3_event("my-bucket", "resume.pdf"), {})
    assert response["statusCode"] == 500
    assert "error" in json.loads(response["body"])
