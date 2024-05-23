
from typing import List, Dict, Any

from base.ai_base import ConfigManager
from base.ai_base import OpenAI_Backend
from base.ai_interface_base import TextInterface

class OpenAI_Text_ConfigManager(ConfigManager):
    def __init__(self,**kwargs):
        super().__init__()
        # Initialize default configurations for chat operations
        self.config = {
            'chat': {
                'model': 'gpt-4',
                'temperature': 0.2
            },
            'embedding': {
                'model': 'text-embedding-ada-002'
            }
        }
        self.update_config(**kwargs)
        

class OpenAITextBackend(OpenAI_Backend,TextInterface):
    
    def __init__(self, api_key: str, config_manager: OpenAI_Text_ConfigManager):
        super().__init__(api_key, config_manager)
        
    def text_chat(self, messages: List[Dict[str, Any]], model: str = 'gpt-4', temperature: float = 0.2) -> str:
        config = self.config_manager.get_config('OpenAI', 'chat')
        params = {**config, 'model': model, 'temperature': temperature}
        try:
            response = self.client.chat.create(
                model=model,
                temperature=temperature,
                messages=messages
            )
            return response.choices[0].message
        except Exception as e:
            self.log_error("OpenAI Chat API error", e)
            return None 
        
    def generate_embedding(self, text: str, model: str = 'text-embedding-ada-002') -> List[float]:
            config = self.config_manager.get_config('OpenAI', 'embedding')
            model = config.get('model', model)
            try:
                response = self.client.embeddings.create(
                    model=model,
                    input=text
                )
                return response.data[0]['embedding']
            except Exception as e:
                self.log_error("OpenAI Embedding API error", e)
                return []