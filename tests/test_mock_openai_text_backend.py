from unittest.mock import Mock, patch

import pytest

from openai_backend.openai_text_backend import OpenAITextBackend

MAX_OUTPUT_TOKENS = 12


@pytest.fixture
def mock_openai_client():
    mock_responses_text = "Hello from responses"
    mock_responses_content = Mock(text=mock_responses_text)
    mock_responses_output = Mock(content=[mock_responses_content])
    mock_responses_response = Mock(output=[mock_responses_output])

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
    mock_embeddings_response.data = [{"embedding": [0.1, 0.2, 0.3]}]

    # Setup the mock client
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_chat_response
    mock_client.responses.create.return_value = mock_responses_response
    mock_client.embeddings.create.return_value = mock_embeddings_response

    mock_client.mock_responses_text = mock_responses_text

    return mock_client


@pytest.fixture
def text_backend(mock_openai_client):
    with patch("openai_backend.openai_text_backend.OpenAITextBackend.create_client", return_value=mock_openai_client):
        backend = OpenAITextBackend()
        yield backend


@pytest.fixture
def messages():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, OpenAI!"},
    ]
    return messages


def test_modern_model_uses_responses(text_backend, mock_openai_client, messages):
    response = text_backend.text_chat(messages, model="gpt-4o")
    assert response == mock_openai_client.mock_responses_text

    mock_openai_client.responses.create.assert_called_once()
    mock_openai_client.chat.completions.create.assert_not_called()

    call_kwargs = mock_openai_client.responses.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["input"] == messages
    assert call_kwargs["temperature"] == text_backend.config_manager.config["chat"]["temperature"]


def test_legacy_model_uses_chat(text_backend, mock_openai_client, messages):
    response = text_backend.text_chat(messages, model="gpt-3.5-turbo")
    assert response == "\n\nHello there, how may I assist you today?"

    mock_openai_client.chat.completions.create.assert_called_once()
    mock_openai_client.responses.create.assert_not_called()

    call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-3.5-turbo"
    assert call_kwargs["messages"] == messages


def test_override_forces_responses(text_backend, mock_openai_client, messages):
    text_backend.text_chat(messages, model="gpt-3.5-turbo", use_responses=True)

    mock_openai_client.responses.create.assert_called_once()
    mock_openai_client.chat.completions.create.assert_not_called()
    assert mock_openai_client.responses.create.call_args.kwargs["model"] == "gpt-3.5-turbo"


def test_override_forces_chat(text_backend, mock_openai_client, messages):
    text_backend.text_chat(messages, model="gpt-5.1", use_responses=False)

    mock_openai_client.chat.completions.create.assert_called_once()
    mock_openai_client.responses.create.assert_not_called()
    assert mock_openai_client.chat.completions.create.call_args.kwargs["model"] == "gpt-5.1"


def test_responses_return_value(text_backend, mock_openai_client, messages):
    response = text_backend.text_chat(messages, model="gpt-4o")
    assert response == mock_openai_client.mock_responses_text

    full_response = text_backend.text_chat(messages, model="gpt-4o", response_type="full")
    assert full_response is mock_openai_client.responses.create.return_value


def test_chat_return_value(text_backend, mock_openai_client, messages):
    response = text_backend.text_chat(messages, model="gpt-3.5-turbo")
    assert response == "\n\nHello there, how may I assist you today?"

    full_response = text_backend.text_chat(messages, model="gpt-3.5-turbo", response_type="full")
    assert full_response is mock_openai_client.chat.completions.create.return_value.choices[0]


def test_max_tokens_maps_to_max_output_tokens(text_backend, mock_openai_client, messages):
    text_backend.text_chat(messages, model="gpt-4o", max_tokens=MAX_OUTPUT_TOKENS)

    call_kwargs = mock_openai_client.responses.create.call_args.kwargs
    assert call_kwargs["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert "max_tokens" not in call_kwargs


def test_text_chat_exception(text_backend, messages):
    with patch.object(text_backend.client.chat.completions, "create", side_effect=Exception("API Error")):
        response = text_backend.text_chat(messages, model="gpt-3.5-turbo")
        assert response is None
