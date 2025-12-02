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
                "size": "1024x1024",
                "quality": "standard",
                "n": 1,
            }
        }
        self.update_config(**kwargs)


class OpenAIImageBackend(ImageInterface, OpenAIBackend):
    def __init__(
        self, api_key: Optional[str] = None, env_var_name: Optional[str] = None, **kwargs: dict[str, Any]
    ) -> None:
        super().__init__(OpenAIImageConfigManager(**kwargs), api_key, env_var_name)

    def generate_image(self, prompt: str, **kwargs: dict[str, Any]) -> dict[str, Optional[str]]:
        config = self.config_manager.combine_config("image_generation", **kwargs)

        try:
            response = self.client.images.generate(prompt=prompt, **config)
            result = response.data[0]
            url = getattr(result, "url", None)
            image = getattr(result, "b64_json", None)
            return {"url": url, "image": image}
        except Exception as e:
            self.log_error("Image generation API error", e)
            raise

    def image_edit(self, image_url: str, edit_options: dict[str, Any], **kwargs: dict[str, Any]) -> Any:
        _ = (image_url, edit_options, kwargs)
        message = "Image edit method not implemented yet"
        raise NotImplementedError(message)

    def image_variation(self, image_url: str, variation_options: dict[str, Any], **kwargs: dict[str, Any]) -> Any:
        _ = (image_url, variation_options, kwargs)
        message = "Generate variation method not implemented yet"
        raise NotImplementedError(message)

    def image_to_text(self, image_url: str, **kwargs: dict[str, Any]) -> Any:
        _ = (image_url, kwargs)
        message = "Image to text method not implemented yet"
        raise NotImplementedError(message)
