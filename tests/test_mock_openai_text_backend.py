from unittest.mock import Mock, patch

import pytest
from openai_backend.openai_text_backend import OpenAITextBackend


@pytest.fixture
def mock_openai_client():
    mock_chat_response = Mock()
    mock_chat_response.choices = [
        Mock(
            index=0,
            message=Mock(role="assistant", content="\n\nHello there, how may I assist you today?"),
            logprobs=None,
            finish_reason="stop",
        )
    ]
    mock_chat_response.usage = {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}

    # Setup the mock response for embeddings
    mock_embeddings_response = Mock()
    mock_embeddings_response.data = [0.1, 0.2, 0.3]

    # Setup the mock client
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_chat_response
    mock_client.embeddings.create.return_value = mock_embeddings_response

    return mock_client


@pytest.fixture
def text_backend(mock_openai_client):
    with patch("openai_backend.openai_text_backend.OpenAITextBackend.create_client", return_value=mock_openai_client):
        backend = OpenAITextBackend()
        yield backend


def test_text_chat_success(text_backend, mock_openai_client):
    # Test successful text chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, OpenAI!"},
    ]

    response = text_backend.text_chat(messages)
    assert response == "\n\nHello there, how may I assist you today?"

    # Verify that the chat.completions.create method was called with the correct arguments
    mock_openai_client.chat.completions.create.assert_called_once_with(
        messages=messages, **text_backend.config_manager.config["chat"]
    )


def test_text_modified_config(text_backend, mock_openai_client):
    # Test successful text chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, OpenAI!"},
    ]

    response = text_backend.text_chat(messages, model="gpt-7", sassiness=11)
    assert response == "\n\nHello there, how may I assist you today?"

    modified_config = text_backend.config_manager.config["chat"].copy()
    modified_config["model"] = "gpt-7"
    modified_config["sassiness"] = 11

    # Verify that the chat.completions.create method was called with the correct arguments
    mock_openai_client.chat.completions.create.assert_called_once_with(messages=messages, **modified_config)


def test_text_chat_exception(text_backend):
    with patch.object(text_backend.client.chat.completions, "create", side_effect=Exception("API Error")):
        # The function in your backend to handle text chat should handle the exception
        response = text_backend.text_chat(["Hello, OpenAI!"])
        assert response is None
