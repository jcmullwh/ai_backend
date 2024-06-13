import unittest

from ai_backend.backend_manager import BackendManager
from openai_backend.openai_audio_backend import OpenAIAudioBackend
from openai_backend.openai_image_backend import OpenAIImageBackend

class TestBackendManager(unittest.TestCase):
    def setUp(self):
        """Set up the test case with a derived ConfigManager."""
        self.backend_manager = BackendManager()

    def test_set_backend(self):
        """Test the set_backend method."""
        # Test case 1: Set backend with valid backend_type and backend_name
        backend_type = "audio"
        backend_name = "openai"
        result, result_name = self.backend_manager.set_backend(backend_type, backend_name)
        
        self.assertIsInstance(result, OpenAIAudioBackend)
        self.assertEqual(result_name, backend_name)
        
        # Test case 2: Set backend with valid backend_type and default backend
        backend_type = "audio"
        result, result_name = self.backend_manager.set_backend(backend_type)
        
        self.assertIsInstance(result, OpenAIAudioBackend)
        self.assertEqual(result_name, backend_name)

        # Test case 3: Set backend with invalid backend_type
        backend_type = "invalid"
        backend_name = "openai"
        with self.assertRaises(ValueError):
            self.backend_manager.set_backend(backend_type, backend_name)

        # Test case 4: Set backend with invalid backend_name
        backend_type = "text"
        backend_name = "invalid"
        with self.assertRaises(ValueError):
            self.backend_manager.set_backend(backend_type, backend_name)
