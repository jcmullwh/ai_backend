from unittest.mock import Mock, patch

import pytest

from src.openai_backend.openai_text_backend import OpenAITextBackend, OpenAITextConfigManager


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
def config_manager():
    # Create an instance of the OpenAITextConfigManager with default settings
    return OpenAITextConfigManager()


@pytest.fixture
def text_backend(mock_openai_client, config_manager):
    # Create an instance of the OpenAITextBackend
    with patch(
        "src.openai_backend.openai_text_backend.OpenAITextBackend.create_client", return_value=mock_openai_client
    ):
        backend = OpenAITextBackend(config_manager=config_manager)
        return backend


def test_text_chat_success(text_backend):
    # Test successful text chat
    response = text_backend.text_chat(["Hello, OpenAI!"])
    assert response == "\n\nHello there, how may I assist you today?"


def test_text_chat_exception(text_backend):
    with patch.object(text_backend.client.chat.completions, "create", side_effect=Exception("API Error")):
        # The function in your backend to handle text chat should handle the exception
        response = text_backend.text_chat(["Hello, OpenAI!"])
        assert response is None
