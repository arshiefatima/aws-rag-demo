"""
Unit tests for query Lambda — run with: pytest tests/ -v
"""
import json
import pytest
from unittest.mock import patch, MagicMock


def make_api_event(question):
    return {"httpMethod": "POST", "body": json.dumps({"question": question})}


@patch.dict("os.environ", {"CHROMA_PATH": "/tmp/test_chroma", "CHROMA_COLLECTION": "test"})
@patch("query.app.generate_answer_ollama")
@patch("query.app.retrieve_context")
def test_handler_returns_answer(mock_retrieve, mock_generate):
    from query.app import handler
    mock_retrieve.return_value = (
        ["Python and LangChain experience"],
        [{"source": "resume.pdf", "page": 1}],
    )
    mock_generate.return_value = "Arshie knows Python and LangChain."

    response = handler(make_api_event("What are the skills?"), {})
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["answer"] == "Arshie knows Python and LangChain."
    assert len(body["sources"]) == 1


@patch.dict("os.environ", {"CHROMA_PATH": "/tmp/test_chroma"})
def test_handler_400_on_empty_question():
    from query.app import handler
    response = handler({"httpMethod": "POST", "body": "{}"}, {})
    assert response["statusCode"] == 400


def test_handler_200_on_options():
    from query.app import handler
    response = handler({"httpMethod": "OPTIONS", "body": None}, {})
    assert response["statusCode"] == 200


@patch.dict("os.environ", {"CHROMA_PATH": "/tmp/test_chroma", "CHROMA_COLLECTION": "test"})
@patch("query.app.retrieve_context")
def test_handler_500_on_error(mock_retrieve):
    from query.app import handler
    mock_retrieve.side_effect = Exception("ChromaDB error")
    response = handler(make_api_event("What are the skills?"), {})
    assert response["statusCode"] == 500
