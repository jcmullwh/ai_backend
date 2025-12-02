import pytest

from ai_backend.api import TextAI
from openai_backend.openai_image_backend import OpenAIImageBackend, OpenAIImageConfigManager

TEMP_OVERRIDE = 0.7


def test_textai_sets_backend_and_delegates(monkeypatch):
    calls: dict[str, object] = {}

    class DummyBackend:
        def text_chat(self, messages, **kwargs):
            calls["messages"] = messages
            calls["kwargs"] = kwargs
            return "ok"

    dummy_backend = DummyBackend()

    def fake_set_backend(_self, _backend_type, selected_backend, _api_key, env_var_name=None, **_kwargs):
        calls["selected_backend"] = selected_backend
        calls["env_var_name"] = env_var_name
        return dummy_backend, selected_backend

    monkeypatch.setattr("ai_backend.api.BackendManager.set_backend", fake_set_backend, raising=True)

    text_ai = TextAI(backend_name="openai", env_var_name="CUSTOM_API_KEY")
    result = text_ai.text_chat([{"role": "user", "content": "hi"}], temperature=TEMP_OVERRIDE)

    assert result == "ok"
    assert calls["selected_backend"] == "openai"
    assert calls["env_var_name"] == "CUSTOM_API_KEY"
    assert calls["messages"][0]["content"] == "hi"
    assert calls["kwargs"]["temperature"] == TEMP_OVERRIDE


def test_image_backend_raises_errors(monkeypatch):
    class DummyImages:
        def generate(self, prompt, **kwargs):  # noqa: ARG002
            error_message = "boom"
            raise Exception(error_message)

    class DummyClient:
        images = DummyImages()

    monkeypatch.setattr(OpenAIImageBackend, "create_client", lambda *_args, **_kwargs: DummyClient(), raising=True)

    backend = OpenAIImageBackend(api_key="test-key")
    with pytest.raises(Exception, match="boom"):
        backend.generate_image("prompt")


def test_image_default_quality_is_supported():
    config = OpenAIImageConfigManager()
    image_cfg = config.get_config("image_generation")
    model = image_cfg.get("model")
    quality = image_cfg.get("quality")

    allowed_by_model = {
        "gpt-image-1": {"low", "medium", "high", "auto"},
        "dall-e-3": {"standard", "hd"},
    }

    assert quality in allowed_by_model.get(model, set())
