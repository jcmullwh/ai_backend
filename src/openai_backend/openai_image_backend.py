from typing import Any, Optional

from base.ai_base import ConfigManager, OpenAIBackend
from base.ai_interface_base import ImageInterface


class OpenAIImageConfigManager(ConfigManager):
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__()
        # Initialize default configurations for image generation operations
        self.config = {
            "image_generation": {
                "model": "dall-e-3",
                "size": "1792x1024",
                "quality": "hd",
                "n": 1,
            }
        }
        self.update_config(**kwargs)


class OpenAIImageBackend(ImageInterface, OpenAIBackend):
    def __init__(self, api_key: Optional[str] = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(OpenAIImageConfigManager(**kwargs), api_key)

    def generate_image(self, prompt: str, **kwargs: dict[str, Any]) -> Any:
        config = self.config_manager.combine_config("image_generation", **kwargs)

        try:
            response = self.client.images.generate(prompt=prompt, **config)
            return response.data[0].url
        except Exception as e:
            self.log_error("Image generation API error", e)
            return None

    def image_edit(self, image_url: str, edit_options: dict[str, Any], **kwargs: dict[str, Any]) -> Any:  # noqa: ARG002
        message = "Image edit method not implemented yet"
        raise NotImplementedError(message)

    def image_variation(self, image_url: str, variation_options: dict[str, Any], **kwargs: dict[str, Any]) -> Any:  # noqa: ARG002
        message = "Generate variation method not implemented yet"
        raise NotImplementedError(message)

    def image_to_text(self, image_url: str, **kwargs: dict[str, Any]) -> Any:  # noqa: ARG002
        message = "Image to text method not implemented yet"
        raise NotImplementedError(message)
