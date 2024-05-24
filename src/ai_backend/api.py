from typing import Any, Optional

from ai_backend.backend_manager import BackendManager


class TextAI:
    def __init__(self, backend_name: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.backend_manager = BackendManager()

        if backend_name:
            self.backend = self.backend_manager.set_backend(backend_name, api_key)

    def text_chat(self, messages: list, **kwargs: dict[str, Any]) -> Any:
        return self.backend.text_chat(messages, **kwargs)

    # Add other methods for image_to_text, image_edit, image_variation, etc.
