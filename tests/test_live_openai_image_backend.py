import pytest

from openai_backend.openai_image_backend import OpenAIImageBackend


@pytest.mark.live_api
def test_live_generate_image():
    text_backend = OpenAIImageBackend()
    prompt = "create an image appropriate for an API image test"
    response = text_backend.generate_image(prompt)
    assert response is not None and isinstance(response, dict)
    assert set(response.keys()) == {"url", "image"}
    assert (response["url"] is None) != (response["image"] is None)


@pytest.mark.live_api
def test_live_generate_image_high_quality_gpt_image():
    text_backend = OpenAIImageBackend()
    prompt = "create a wide image appropriate for an API image test"
    response = text_backend.generate_image(
        prompt,
        model="gpt-image-1",
        size="1536x1024",
        quality="high",
    )
    assert response is not None and isinstance(response, dict)
    assert set(response.keys()) == {"url", "image"}
    assert response["url"] is None
    assert isinstance(response["image"], str)
    assert len(response["image"]) > 0
