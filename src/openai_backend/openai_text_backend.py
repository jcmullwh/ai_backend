from typing import Any, Optional

from base.ai_base import ConfigManager, OpenAIBackend
from base.ai_interface_base import TextInterface


class OpenAITextConfigManager(ConfigManager):
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__()
        # Initialize default configurations for chat operations
        self.config = {
            "chat": {"model": "gpt-4o", "temperature": 0.2},
            "embedding": {"model": "text-embedding-ada-002"},
        }
        self.update_config(**kwargs)


class OpenAITextBackend(OpenAIBackend, TextInterface):
    def __init__(self, api_key: Optional[str] = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(OpenAITextConfigManager(**kwargs), api_key)

    def text_chat(self, messages: list, response_type: Optional[str] = None, **kwargs: dict[str, Any]) -> Any:
        config = self.config_manager.combine_config("chat", **kwargs)

        try:
            response = self.client.chat.completions.create(messages=messages, **config)
            if response_type == "full":
                return response.choices[0]
            else:
                return response.choices[0].message.content
        except Exception as e:
            self.log_error("OpenAI Chat API error", e)
            return None

    def generate_embedding(self, messages: list, **kwargs: dict[str, Any]) -> Any:
        config = self.config_manager.combine_config("embedding", **kwargs)
        model = config.get("model")
        try:
            response = self.client.embeddings.create(model=model, input=messages, **config)
            return response.data[0]["embedding"]
        except Exception as e:
            self.log_error("OpenAI Embedding API error", e)
            return None
