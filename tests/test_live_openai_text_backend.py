import pytest
from openai_backend.openai_text_backend import OpenAITextBackend


@pytest.mark.live_api
def test_live_text_chat():
    text_backend = OpenAITextBackend()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, OpenAI!"},
    ]
    response = text_backend.text_chat(messages)
    # Check if the response is not None and is a string
    assert response is not None and isinstance(response, str)


"""
@pytest.mark.live_api
def test_live_generate_embedding():
    config_manager = OpenAITextConfigManager()
    text_backend = OpenAITextBackend(api_key="your_real_api_key", config_manager=config_manager)
    response = text_backend.generate_embedding(["Hello, OpenAI!"])
    assert response is not None and isinstance(response, list)
"""
