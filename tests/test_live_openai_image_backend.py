import pytest
from openai_backend.openai_image_backend import OpenAIImageBackend


@pytest.mark.live_api
def test_live_generate_image():
    text_backend = OpenAIImageBackend()
    prompt = "create an image appropriate for an API image test"
    response = text_backend.generate_image(prompt)
    # Check if the response is not None and is a string
    assert response is not None and isinstance(response, str)
