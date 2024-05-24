# base/backend_manager.py
from typing import Any, Optional

from base.ai_interface_base import TextInterface
from openai_backend.openai_text_backend import OpenAITextBackend


class BackendManager:
    def __init__(self) -> None:
        self.backends: dict[str, Any] = {
            "openai": OpenAITextBackend,
            # Add other backends here
        }
        # Default Backend
        self.current_backend: TextInterface = OpenAITextBackend()

    def set_backend(self, backend_name: str, api_key: Optional[str] = None) -> Any:
        if backend_name in self.backends:
            backend_class = self.backends[backend_name]
            return backend_class(api_key)
        else:
            error_message = f"Backend {backend_name} not supported."
            raise ValueError(error_message)

    def get_backend(self) -> TextInterface:
        if not self.current_backend:
            error_message = "No backend is currently set."
            raise ValueError(error_message)
        else:
            return self.current_backend
