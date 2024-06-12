import pytest
from ai_backend import ImageAI, TextAI


@pytest.mark.live_api
def test_live_text_chat():
    text_ai = TextAI(backend_name="openai")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "This is a test message. Please respond with 'Test response'."},
    ]
    response = text_ai.text_chat(messages)
    # Check if the response is not None and is a string
    assert response is not None and isinstance(response, str)


@pytest.mark.live_api
def test_live_text_chat_modified_config():
    text_ai = TextAI(backend_name="openai")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "This is a test message. Please respond with 'Test response'."},
    ]
    response = text_ai.text_chat(messages, temperature=0.5, max_tokens=50)
    # Check if the response is not None and is a string
    assert response is not None and isinstance(response, str)


@pytest.mark.live_api
def test_live_generate_image():
    image_ai = ImageAI(backend_name="openai")

    prompt = "create an image appropriate for an API image test"
    response = image_ai.generate_image(prompt)
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
