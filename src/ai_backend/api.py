import io
from typing import Any, Optional, Union

from ai_backend.backend_manager import BackendManager


class TextAI:
    def __init__(self, backend_name: Optional[str] = None, api_key: Optional[str] = None) -> None:
        """Initialize a TextAI instance with an optional backend and API key.
        If no backend is specified, the default backend is used.
        If no API key is specified, it is retrieved from the environment variables.

        Args:
            backend_name (Optional[str]): The name of the backend to use.
                If None, the default backend is used.
            api_key (Optional[str]): The API key for accessing the specified backend.
                If None, it attempts to retrieve from the environment variables.
        """
        self.backend_manager = BackendManager()
        self.backend_type = "text"
        if backend_name:
            self.backend = self.backend_manager.set_backend(self.backend_type, backend_name, api_key)

    def text_chat(self, messages: list, **kwargs: dict[str, Any]) -> Any:
        """Send messages to the backend for text-based chatting.

        Args:
            messages (list): A list of messages for the chat.
            **kwargs (dict[str, Any]): Additional keyword arguments specific to the backend's chat function.

        Returns:
            Any: The response from the backend.
        """
        return self.backend.text_chat(messages, **kwargs)


class ImageAI:
    def __init__(self, backend_name: Optional[str] = None, api_key: Optional[str] = None) -> None:
        """Initialize an ImageAI instance with an optional backend and API key.
        If no backend is specified, the default backend is used.
        If no API key is specified, it is retrieved from the environment variables.

        Args:
            backend_name (Optional[str]): The name of the backend to use.
                If None, the default backend is used.
            api_key (Optional[str]): The API key for accessing the specified backend.
                If None, it attempts to retrieve from the environment variables.
        """
        self.backend_manager = BackendManager()
        self.backend_type = "image"

        if backend_name:
            self.backend = self.backend_manager.set_backend(self.backend_type, backend_name, api_key)

    def generate_image(self, messages: list, **kwargs: dict[str, Any]) -> Any:
        """Generate images based on the provided messages.

        Args:
            messages (list): Input data for image generation, usually text prompts.
            **kwargs (dict[str, Any]): Additional parameters for the backend's image generation function.

        Returns:
            Any: The generated images from the backend.
        """
        return self.backend.generate_image(messages, **kwargs)


class AudioAI:
    def __init__(self, backend_name: Optional[str] = None, api_key: Optional[str] = None) -> None:
        """Initialize an AudioAI instance with an optional backend and API key.
        If no backend is specified, the default backend is used.
        If no API key is specified, it is retrieved from the environment variables.

        Args:
            backend_name (Optional[str]): The name of the backend to use.
                If None, the default backend is used.
            api_key (Optional[str]): The API key for accessing the specified backend.
                If None, it attempts to retrieve from the environment variables.
        """
        self.backend_manager = BackendManager()
        self.backend_type = "audio"

        if backend_name:
            self.backend = self.backend_manager.set_backend(self.backend_type, backend_name, api_key)

    def voice_to_text(
        self,
        audio_input: Union[bytes, io.BufferedReader],
        **kwargs: Any,
    ) -> Any:
        """Convert voice messages to text using the backend's capabilities.

        Args:
            messages (list): Audio files or streams to be converted.
            **kwargs (dict[str, Any]): Additional parameters for the backend's voice-to-text function.

        Returns:
            Any: The textual representation of the spoken content.
        """
        return self.backend.voice_to_text(audio_input, **kwargs)
