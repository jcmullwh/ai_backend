from typing import Any, Optional

from base.ai_base import ConfigManager, OpenAIBackend
from base.ai_interface_base import TextInterface


class OpenAITextConfigManager(ConfigManager):
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__()
        # Initialize default configurations for chat operations
        self.config = {
            "chat": {"model": "gpt-5.2", "temperature": 0.2},
            "embedding": {"model": "text-embedding-3-large"},
        }
        self.update_config(**kwargs)


class OpenAITextBackend(OpenAIBackend, TextInterface):
    def __init__(
        self, api_key: Optional[str] = None, env_var_name: Optional[str] = None, **kwargs: dict[str, Any]
    ) -> None:
        super().__init__(OpenAITextConfigManager(**kwargs), api_key, env_var_name)

    def _should_use_responses(self, model: str, *, use_responses_override: Optional[bool] = None) -> bool:
        """
        Decide whether to call the Responses API or Chat Completions.

        Rules:
        - If use_responses_override is True -> Responses.
        - If use_responses_override is False -> Chat.
        - Otherwise:
            - If model starts with 'gpt-4' or 'gpt-5' -> Responses.
            - Else -> Chat.
        """
        if use_responses_override is True:
            return True
        if use_responses_override is False:
            return False
        return model.startswith(("gpt-4", "gpt-5"))

    def _to_responses_input(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert a standard chat `messages` list into Responses `input`.

        Today we assume each message is:
            {"role": <str>, "content": <str>}
        and we simply reuse this structure, which is accepted by Responses.

        This helper exists so we can later extend it to handle multimodal content
        without changing the rest of `text_chat`.
        """
        responses_input: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                error_message = "Each message must be a dict with 'role' and 'content'."
                raise ValueError(error_message)
            role = message.get("role")
            content = message.get("content")
            if role is None or content is None:
                error_message = "Each message must contain 'role' and 'content'."
                raise ValueError(error_message)
            if not isinstance(content, str):
                error_message = "Message content must be a string for the Responses API."
                raise ValueError(error_message)
            responses_input.append({"role": role, "content": content})
        return responses_input

    def text_chat(self, messages: list, response_type: Optional[str] = None, **kwargs: dict[str, Any]) -> Any:
        """
        Chat with the model using either the Responses API (preferred for modern models)
        or Chat Completions (fallback for legacy models).

        - messages: list of {"role": ..., "content": str}
        - response_type: "full" returns the first choice/block; otherwise returns text.
        - kwargs: overrides for config, plus:
            - use_responses: Optional[bool] -> force/forbid Responses.
        """
        try:
            config = self.config_manager.combine_config("chat", **kwargs)
        except Exception as e:
            self.log_error("Error combining chat config", e)
            return None

        model = config.pop("model", None)
        if model is None:
            self.log_error("No 'model' specified in chat config", ValueError("missing model"))
            return None

        try:
            use_responses_override = config.pop("use_responses", None)
            use_responses = self._should_use_responses(model, use_responses_override=use_responses_override)

            if use_responses:
                responses_input = self._to_responses_input(messages)
                payload: dict[str, Any] = {
                    "model": model,
                    "input": responses_input,
                }

                if "max_tokens" in config and "max_output_tokens" not in config:
                    payload["max_output_tokens"] = config.pop("max_tokens")

                payload.update(config)

                response = self.client.responses.create(**payload)

                if response_type == "full":
                    return response

                try:
                    return response.output[0].content[0].text
                except Exception as e:
                    self.log_error("Failed to parse Responses output", e)
                    return None

            payload = {"model": model, "messages": messages}
            payload.update(config)
            response = self.client.chat.completions.create(**payload)

            if response_type == "full":
                return response.choices[0]
            return response.choices[0].message.content
        except Exception as e:
            self.log_error("Error in text_chat", e)
            return None

    def generate_embedding(self, messages: list, **kwargs: dict[str, Any]) -> Any:
        config = self.config_manager.combine_config("embedding", **kwargs)
        model = config.get("model")
        try:
            response = self.client.embeddings.create(model=model, input=messages, **config)
            return response.data[0]["embedding"]
        except Exception as e:
            self.log_error("OpenAI Embedding API error", e)
            return None
