import io
from abc import ABC, abstractmethod
from typing import Any, Optional, Union


class AudioInterface(ABC):
    @abstractmethod
    def voice_to_text(
        self, audio_input: Union[bytes, io.BufferedReader], chunk_length: int, overlap: int, **kwargs: dict[str, Any]
    ) -> Any:
        """
        Transcribes voice or audio input into text.

        Parameters:
            audio_input (Union[bytes, io.BufferedReader]): The audio data to be transcribed.
                This can be raw byte data or a data stream.
            **kwargs: Additional keyword arguments to customize the transcription process,
                such as specifying the language, dialect, or any model-specific parameters.

        Returns:
            str: The transcribed text from the audio input.
        """
        pass

    @abstractmethod
    def text_to_speech(self, text: str, **kwargs: dict[str, Any]) -> Any:
        """
        Converts text into spoken audio.

        Parameters:
            text (str): The text to be converted into speech.
            **kwargs: Additional keyword arguments to customize the speech synthesis process,
                such as voice characteristics (gender, age, accent), speech rate, and volume.

        Returns:
            Union[bytes, io.BytesIO]: The generated speech audio as bytes or a data stream,
                depending on the preferences specified in kwargs.
        """
        pass

    @abstractmethod
    def audio_chat(self, audio_input: Union[bytes, io.BufferedReader], **kwargs: dict[str, Any]) -> Any:
        """
        Processes an audio input through a chat interaction and returns a textual response.

        Parameters:
            audio_input (Union[bytes, io.BufferedReader]): The audio data representing spoken input for the chat.
            **kwargs: Additional keyword arguments to customize the processing, such as context, language models,
                      or any specific settings for handling the audio-chat interaction.

        Returns:
            str: The textual response from processing the audio chat interaction.
        """
        pass


class TextInterface(ABC):
    @abstractmethod
    def text_chat(self, messages: list, response_type: Optional[str] = None, **kwargs: dict[str, Any]) -> Any:
        """
        Processes a chat interaction based on a list of messages.

        Parameters:
            messages (list): A list of messages, where each message could be a string or a structured object.
            **kwargs: Additional keyword arguments for more customization.

        Returns:
            str: The response text from the chat interaction.
        """
        pass

    @abstractmethod
    def generate_embedding(self, messages: list, **kwargs: dict[str, Any]) -> Any:
        """
        Generates an embedding for a list of messages.

        Parameters:
            messages (list): A list of messages to be used for generating the embedding.
            **kwargs: Additional keyword arguments for more customization.

        Returns:
            str: A representation of the embedding, could be a string or a URL/path to the embedding data.
        """
        pass


class ImageInterface(ABC):
    @abstractmethod
    def generate_image(self, prompt: str, **kwargs: dict[str, Any]) -> Any:
        """
        Generates an image based on a given textual prompt.

        Parameters:
            prompt (str): A textual description or prompt that specifies the content or theme of the image to be
                          generated.
            **kwargs: Additional keyword arguments to customize the image generation process, such as the style,
                      resolution, and specific parameters for the generation model being used.

        Returns:
            str: A URL or path to the generated image.
        """
        pass

    @abstractmethod
    def image_to_text(self, image_url: str, **kwargs: dict[str, Any]) -> Any:
        """
        Converts the content of an image into a textual description or extracts textual information from the image.

        Parameters:
            image_data (Union[bytes, io.BytesIO, str]): The image data to be processed. This can be a byte stream,
                                                        a file-like object, or a path to the image file.
            **kwargs: Additional keyword arguments to customize the image-to-text conversion process, such as the
                model used for analysis, language settings, or specific attributes to focus on during the analysis.

        Returns:
            str: The textual description or extracted text from the image.
        """
        pass

    @abstractmethod
    def image_edit(self, image_url: str, edit_options: dict[str, Any], **kwargs: dict[str, Any]) -> Any:
        """
        Edits an image based on specified options.

        Parameters:
            image_data (Union[bytes, io.BytesIO, str]): The original image data to be edited.
            edit_options (dict): A dictionary specifying the editing parameters.
            **kwargs: Additional keyword arguments for more customization.

        Returns:
            str: A URL or path to the edited image.
        """
        pass

    @abstractmethod
    def image_variation(self, image_url: str, variation_options: dict[str, Any], **kwargs: dict[str, Any]) -> Any:
        """
        Generates variations of a given image based on specified options.

        Parameters:
            image_data (Union[bytes, io.BytesIO, str]): The original image data for which variations are generated.
            variation_options (dict): A dictionary specifying the variation parameters.
            **kwargs: Additional keyword arguments for more customization.

        Returns:
            str: A URL or path to the generated image variations.
        """
        pass
