import unittest
from typing import Any, Dict
from src.openai.openai_audio_backend import OpenAI_Audio_ConfigManager


class TestConfigManager(unittest.TestCase):
    
    def setUp(self):
        """Set up the test case with a derived ConfigManager."""
        self.config_manager = OpenAI_Audio_ConfigManager(
            transcription={'model': 'whisper-2', 'speed': '2x'},
            text_to_speech={'voice': 'voice3'}
        )

    def test_get_config(self):
        """Test retrieving existing and non-existing configurations."""
        transcription_config = self.config_manager.get_config('transcription')
        self.assertEqual(transcription_config['model'], 'whisper-2')
        self.assertEqual(transcription_config['speed'], '2x')

        non_existing_config = self.config_manager.get_config('non_existing')
        self.assertEqual(non_existing_config, {})

    def test_update_config(self):
        """Test updating existing and new configurations."""
        # Update existing configuration
        self.config_manager.update_config(transcription={'response_format': 'simple_json'})
        transcription_config = self.config_manager.get_config('transcription')
        self.assertEqual(transcription_config['response_format'], 'simple_json')

        # Update with new configuration
        self.config_manager.update_config(new_service={'param1': 'value1'})
        new_service_config = self.config_manager.get_config('new_service')
        self.assertEqual(new_service_config['param1'], 'value1')

        # Invalid type update
        with self.assertRaises(TypeError):
            self.config_manager.update_config(transcription=123)

    def test_combine_config(self):
        """Test combining configurations."""
        combined_config = self.config_manager.combine_config(
            'transcription',
            model='whisper-3',
            speed='1.5x',
            new_param='new_value'
        )
        self.assertEqual(combined_config['model'], 'whisper-3')
        self.assertEqual(combined_config['speed'], '1.5x')
        self.assertEqual(combined_config['new_param'], 'new_value')
        self.assertEqual(combined_config['response_format'], 'verbose_json')  # Ensure existing config is preserved

        with self.assertRaises(ValueError):
            self.config_manager.combine_config('non_existing_service', model='model-x')

if __name__ == '__main__':
    unittest.main()
