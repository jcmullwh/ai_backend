# base/backend_manager.py
from typing import Any, Optional

from openai_backend.openai_audio_backend import OpenAIAudioBackend
from openai_backend.openai_image_backend import OpenAIImageBackend
from openai_backend.openai_text_backend import OpenAITextBackend


class BackendManager:
    def __init__(self) -> None:
        self.backends: dict[str, dict[str, Any]] = {
            "text": {
                "openai": OpenAITextBackend,
            },
            "image": {
                "openai": OpenAIImageBackend,
            },
            "audio": {
                "openai": OpenAIAudioBackend,
            },
        }
        self.default_backend: dict[str, str] = {
            "text": "openai",
            "image": "openai",
            "audio": "openai",
        }

    def set_backend(self, backend_type: str, backend_name: str, api_key: Optional[str] = None) -> Any:

        if backend_type in self.backends:
            if backend_name in self.backends[backend_type]:
                backend_class = self.backends[backend_type][backend_name]
                self.current_backend = backend_class(api_key)
                return backend_class(api_key)
            elif backend_name is None:
                backend_name = self.default_backend[backend_type]
                backend_class = self.backends[backend_name]
                self.current_backend = backend_class(api_key)
            else:
                error_message = f"Backend {backend_name} not supported for Backend Type {backend_type}."
                raise ValueError(error_message)
        else:
            error_message = f"Backend Type {backend_type} not supported."
            raise ValueError(error_message)

    def get_backend(self) -> Any:
        if not self.current_backend:
            error_message = "No backend is currently set."
            raise ValueError(error_message)
        else:
            return self.current_backend
