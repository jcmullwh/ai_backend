from typing import Any

from base.ai_base import ConfigManager, OpenAIBackend
from base.ai_interface_base import ImageInterface


class OpenAIImageConfigManager(ConfigManager):
    def __init__(self) -> None:
        super().__init__()
        # Initialize default configurations for image generation operations
        self.config = {"image_generation": {"model": "dall-e-3", "size": "1024x1792", "quality": "hd"}}


class OpenAIImageBackend(ImageInterface, OpenAIBackend):
    def __init__(self, api_key: str, config_manager: OpenAIImageConfigManager) -> None:
        super().__init__(api_key, config_manager)

    def generate_image(self, prompt: str, **kwargs: dict[str, Any]) -> Any:
        config = self.config_manager.combine_config("image_generation", **kwargs)

        try:
            response = self.client.images.generate(
                model=config["model"],
                prompt=prompt,
                size=config["size"],
                quality=config["quality"],
                style=config["style"],
                n=config["num_images"],
            )
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
