import io
import logging
from typing import Any, Optional, Union

from base.ai_base import ConfigManager, OpenAIBackend
from base.ai_interface_base import AudioInterface
from fuzzywuzzy import fuzz  # type: ignore
from pydub import AudioSegment  # type: ignore

logger = logging.getLogger(__name__)


class OpenAIAudioConfigManager(ConfigManager):
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.config = {
            "transcription": {"model": "whisper-1", "response_format": "verbose_json", "timestamps": ["segment"]},
            "text_to_speech": {"model": "tts-model", "voice": "default-voice"},
        }

        self.update_config(**kwargs)


class OpenAIAudioBackend(AudioInterface, OpenAIBackend):
    def __init__(self, api_key: str, config_manager: OpenAIAudioConfigManager) -> None:
        super().__init__(api_key, config_manager)

    def voice_to_text(
        self,
        audio_input: Union[bytes, io.BufferedReader],
        chunk_length: int = 600000,
        overlap: int = 5000,
        **kwargs: Any,
    ) -> Any:
        config = self.config_manager.combine_config("transcription", **kwargs)

        buffer = io.BytesIO()
        if isinstance(audio_input, bytes):
            buffer.write(audio_input)
        elif isinstance(audio_input, io.BytesIO):
            buffer = audio_input
        else:
            error = "Unsupported audio input type."
            raise ValueError(error)

        buffer.seek(0)
        audio = AudioSegment.from_file(buffer)
        chunks = [audio[i : i + chunk_length + overlap] for i in range(0, len(audio), chunk_length - overlap)]

        transcriptions: list[str] = []
        for chunk in chunks:
            transcription = self.process_chunk(chunk, config)
            if transcription:
                transcriptions.append(transcription)

        character_overlap = overlap / 1000 * 16 * 5
        full_transcription = self.stitch_transcriptions(transcriptions, character_overlap)
        return full_transcription

    def process_chunk(self, chunk: Any, config: dict[str, Any]) -> Any:
        buffer = io.BytesIO()
        chunk.export(buffer, format="mp3")
        buffer.seek(0)

        try:
            response = self.client.audio.transcriptions.create(
                file=("filename.mp3", buffer, "audio/mpeg"),
                model=config["model"],
                response_format=config["response_format"],
                timestamp_granularities=config["timestamps"],
            )
            return response.text
        except Exception as e:
            logger.error(f"Audio transcription API error: {e!s}")
            return None
        finally:
            buffer.close()

    def find_best_overlap(self, stitched_text: str, current_text: str, overlap: int = 200) -> int:
        best_ratio = 0
        best_index = -1
        max_length = min(int(round(overlap)), len(stitched_text), len(current_text))

        for i in range(50, max_length):
            ratio = fuzz.partial_ratio(stitched_text[-i:], current_text[:i])
            if ratio > best_ratio:
                best_ratio = ratio
                best_index = len(stitched_text) - i

        return best_index

    def stitch_transcriptions(self, transcriptions: list[str], overlap: float = 5000) -> str:
        stitched_text = transcriptions[0]
        for current_text in transcriptions[1:]:
            overlap_index = self.find_best_overlap(stitched_text, current_text, int(overlap))
            stitched_text = (
                stitched_text[:overlap_index] + current_text if overlap_index != -1 else stitched_text + current_text
            )

        return stitched_text.strip()

    def text_to_speech(self, text: str, **kwargs: Any) -> Optional[Union[bytes, io.BytesIO]]:
        try:
            response = self.client.audio.speech.create(
                model=kwargs.get("model", "tts-model"), voice=kwargs.get("voice", "default-voice"), input=text
            )
            return io.BytesIO(response.audio)
        except Exception as e:
            logger.error(f"Text-to-speech API error: {e!s}")
            return None

    def audio_chat(self, audio_input: Union[bytes, io.BufferedReader], **kwargs: Any) -> Any:
        transcribed_text = self.voice_to_text(audio_input, **kwargs)
        if transcribed_text:
            return transcribed_text
        return None
