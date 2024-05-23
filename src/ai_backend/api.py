from typing import Any

from base.ai_base import ConfigManager

from ai_backend.backend_manager import BackendManager


class API:
    def __init__(self, backend_name: str, api_key: str) -> None:
        self.config_manager = ConfigManager()
        self.backend_manager = BackendManager()
        self.backend_manager.set_backend(backend_name, api_key, self.config_manager)

    def generate_image(self, prompt: str, **kwargs: dict[str, Any]) -> Any:
        backend = self.backend_manager.get_backend()
        return backend.generate_image(prompt, **kwargs)

    # Add other methods for image_to_text, image_edit, image_variation, etc.
