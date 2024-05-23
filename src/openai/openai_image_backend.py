from typing import Any, Optional
from base.ai_base import ConfigManager
from base.ai_base import OpenAI_Backend
from base.ai_interface_base import ImageInterface

class OpenAI_Image_ConfigManager(ConfigManager):
    def __init__(self):
        super().__init__()
        # Initialize default configurations for image generation operations
        self.config = {
            'image_generation': {
                'model': 'dall-e-3',
                'size': '1024x1792',
                'quality': 'hd'
            }
        }


class OpenAI_Image_Backend(ImageInterface,OpenAI_Backend):
    
    def __init__(self, api_key: str, config_manager: OpenAI_Image_ConfigManager):
        super().__init__(api_key, config_manager)
        
        
    def generate_image(self, prompt: str, **kwargs: Any) -> Optional[str]:
        config = self.config_manager.get_config('image_generation')
        params = {**config, **kwargs}
        try:
            response = self.client.images.generate(model=params['model'], prompt=prompt, size=params['size'], quality=params['quality'], style=params['style'], n=params['num_images'])
            return response.data[0].url
        except Exception as e:
            self.log_error("Image generation API error",e)
            return None    