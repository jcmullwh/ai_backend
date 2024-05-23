# base/backend_manager.py

from typing import Dict, Type
from base.ai_interface_base import ImageInterface
from openai.openai_image_backend import OpenAI_Image_Backend
from base.ai_base import ConfigManager

class BackendManager:
    def __init__(self):
        self.backends: Dict[str, Type[ImageInterface]] = {
            'openai': OpenAI_Image_Backend,
            # Add other backends here
        }
        self.current_backend: ImageInterface = None
    
    def set_backend(self, backend_name: str, api_key: str, config_manager: ConfigManager):
        if backend_name in self.backends:
            backend_class = self.backends[backend_name]
            self.current_backend = backend_class(api_key, config_manager)
        else:
            raise ValueError(f"Backend {backend_name} not supported.")
    
    def get_backend(self) -> ImageInterface:
        if not self.current_backend:
            raise ValueError("No backend is currently set.")
        return self.current_backend