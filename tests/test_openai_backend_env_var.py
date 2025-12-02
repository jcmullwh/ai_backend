from openai_backend.openai_text_backend import OpenAITextBackend


def test_openai_backend_uses_default_env_var(monkeypatch):
    api_key = "default-env-key"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)

    captured: dict[str, str] = {}

    def fake_create_client(_self, key):
        captured["api_key"] = key
        return object()

    monkeypatch.setattr(OpenAITextBackend, "create_client", fake_create_client, raising=True)

    OpenAITextBackend(api_key=None)

    assert captured["api_key"] == api_key


def test_openai_backend_uses_custom_env_var(monkeypatch):
    api_key = "custom-env-key"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_OPENAI_KEY", api_key)

    captured: dict[str, str] = {}

    def fake_create_client(_self, key):
        captured["api_key"] = key
        return object()

    monkeypatch.setattr(OpenAITextBackend, "create_client", fake_create_client, raising=True)

    OpenAITextBackend(api_key=None, env_var_name="CUSTOM_OPENAI_KEY")

    assert captured["api_key"] == api_key
