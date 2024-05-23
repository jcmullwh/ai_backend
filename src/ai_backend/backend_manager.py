# base/backend_manager.py
from typing import Any

from base.ai_base import ConfigManager
from base.ai_interface_base import ImageInterface
from openai_backend.openai_image_backend import OpenAIImageBackend, OpenAIImageConfigManager


class BackendManager:
    def __init__(self) -> None:
        self.backends: dict[str, Any] = {
            "openai": OpenAIImageBackend,
            # Add other backends here
        }
        self.current_backend: ImageInterface = OpenAIImageBackend("", OpenAIImageConfigManager())

    def set_backend(self, backend_name: str, api_key: str, config_manager: ConfigManager) -> None:
        if backend_name in self.backends:
            backend_class = self.backends[backend_name]
            self.current_backend = backend_class(api_key, config_manager)
        else:
            error_message = f"Backend {backend_name} not supported."
            raise ValueError(error_message)

    def get_backend(self) -> ImageInterface:
        if not self.current_backend:
            error_message = "No backend is currently set."
            raise ValueError(error_message)
        else:
            return self.current_backend
