import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from openai import Client, OpenAI

logger = logging.getLogger(__name__)


class ConfigManager:
    def __init__(self) -> None:
        # Initialize the configuration dictionary. This dictionary will store configurations for different services.
        self.config: dict[str, dict[str, Any]] = {}

        # Setup logging for the ConfigManager. This will help in debugging and logging errors or important information.
        self.logger = logging.getLogger("ConfigManager")
        self.logger.setLevel(logging.DEBUG)  # Set logging level to DEBUG to capture all types of log messages
        handler = logging.StreamHandler()  # Create a stream handler to output logs to the console
        handler.setLevel(logging.DEBUG)  # Set the level for the handler
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )  # Define the log message format
        handler.setFormatter(formatter)  # Set the formatter for the handler
        self.logger.addHandler(handler)  # Add the handler to the logger

    def get_config(self, service: str) -> dict[str, Any]:
        """
        Retrieve the configuration for a specific service.

        Args:
            service (str): The name of the service to retrieve the configuration for.

        Returns:
            dict[str, Any]: The configuration dictionary for the specified service.
                Returns an empty dictionary if the service is not found.
        """
        return self.config.get(service, {})

    def update_config(self, **kwargs: dict[str, Any]) -> None:
        """
        Update the configuration for one or more services.

        Args:
            kwargs (dict[str, Any]): Keyword arguments where the key is the service name and the value
                is the configuration dictionary.

        Raises:
            TypeError: If the updates provided are not in the expected format (dict or str).
        """
        for service, updates in kwargs.items():
            if service in self.config:
                # If the service already has a configuration, update it with the new values.
                if isinstance(updates, dict):
                    self._update_nested_dict(self.config[service], updates)
                elif isinstance(updates, str):
                    # Allow setting the entire configuration to a string value directly.
                    self.config[service] = updates
                else:
                    # Log and raise an error if the updates are not of the expected type.
                    error_message = f"Invalid type for '{service}'. Expected dict or str, got {type(updates).__name__}."
                    self.logger.error(error_message)
                    raise TypeError(error_message)
            elif isinstance(updates, dict):
                self.config[service] = updates
            elif isinstance(updates, str):
                # Allow setting a new configuration to a string value directly.
                self.config[service] = updates
            else:
                # Log and raise an error if the updates are not of the expected type.
                error_message = f"Invalid type for new configuration '{service}'. \
                    Expected dict or str, got {type(updates).__name__}."
                self.logger.error(error_message)
                raise TypeError(error_message)

    def _update_nested_dict(self, target: dict[str, Any], updates: dict[str, Any]) -> None:
        """
        Recursively update a nested dictionary with new values.

        Args:
            target (dict[str, Any]): The original dictionary to be updated.
            updates (dict[str, Any]): The dictionary with new values to update the original dictionary with.
        """
        for key, value in updates.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # If both target and updates have dictionaries for the same key,
                # update the nested dictionary recursively.
                self._update_nested_dict(target[key], value)
            else:
                # Otherwise, simply set the value in the target dictionary.
                target[key] = value

    def set_default(self, service: str, **kwargs: dict[str, Any]) -> None:
        """
        Change default configuration for a service.

        Args:
            service (str): The name of the service.
            value (dict[str, Any]): The default configuration dictionary for the service.

        Raises:
            TypeError: If the provided value is not a dictionary.
        """
        if service is None:
            error_message = "Service name cannot be None."
            self.logger.error(error_message)
            raise ValueError(error_message)

        self.config[service] = self.combine_config(service, **kwargs)

    def add_default(self, service: str, params: dict[str, Any]) -> None:
        """
        Add a new default parameter for a service.

        Args:
            service (str): The name of the service.
            value (dict[str, Any]): The default configuration dictionary for the service.

        Raises:
            TypeError: If the provided value is not a dictionary.
            ValueError: If a configuration for the service already exists.
        """
        # if service is none, raise an error
        if service is None:
            error_message = "Service name cannot be None."
            self.logger.error(error_message)
            raise ValueError(error_message)

        for key, value in params.items():
            if key not in self.config[service]:
                self.config[service][key] = value
            else:
                # Log and raise an error if the value is not a dictionary.
                error_message = f"{key} already exists in '{service}'."
                self.logger.error(error_message)
                raise TypeError(error_message)

    def add_service(self, service: str, params: dict[str, Any]) -> None:
        """
        Add a new service configuration.

        Args:
            service (str): The name of the service.
            value (dict[str, Any]): The default configuration dictionary for the service.

        Raises:
            TypeError: If the provided value is not a dictionary.
            ValueError: If a configuration for the service already exists.
        """
        # if service is none, raise an error
        if service is None:
            error_message = "Service name cannot be None."
            self.logger.error(error_message)
            raise ValueError(error_message)

        if service not in self.config:
            self.config[service] = params
        else:
            # Log and raise an error if the service already has a configuration.
            error_message = f"Configuration for '{service}' already exists."
            self.logger.error(error_message)
            raise ValueError(error_message)

    def combine_config(self, service: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Combine the default configuration with the provided configuration.

        Args:
            service (str): The name of the service.
            kwargs (Any): Keyword arguments representing the new configuration values
                to be combined with the default configuration.

        Returns:
            dict[str, Any]: The combined configuration dictionary.

        Raises:
            ValueError: If the service does not exist in the default configurations.
            TypeError: If the provided configuration values are not in the expected format.
        """
        if service not in self.config:
            # Log and raise an error if the service is not found in the default configurations.
            error_message = f"Service '{service}' not found in default configurations."
            self.logger.error(error_message)
            raise ValueError(error_message)

        # Create a copy of the default configuration to avoid modifying the original.
        config = self.config[service].copy()

        # Iterate over the provided configuration values and update the default configuration.
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self._update_nested_dict(config, {key: value})
            else:
                config[key] = value

        return config


class AIBackend(ABC):
    def __init__(self, config_manager: ConfigManager, api_key: Optional[str]) -> None:
        if not isinstance(config_manager, ConfigManager):
            config_manager_error = "config_manager must be an instance of ConfigManager"
            raise TypeError(config_manager_error)
        if not isinstance(api_key, (str, type(None))):
            api_key_error = "api_key must be a string or None"
            raise TypeError(api_key_error)

        if not api_key:
            env_var_name = self.get_env_var_name()
            api_key = os.getenv(env_var_name)
            if not api_key:
                error = f"{env_var_name} is not set. You need to set this environment \
                    variable or provide the API key as an argument."
                logger.error(error)
                raise ValueError(error)

        self.client = self.create_client(api_key)
        self.config_manager = config_manager

    def set_default(self, service: str, **kwargs: dict[str, Any]) -> None:
        self.config_manager.set_default(service, **kwargs)

    def add_default(self, service: str, **kwargs: dict[str, Any]) -> None:
        self.config_manager.add_default(service, **kwargs)

    def add_service(self, service: str, config: dict[str, Any]) -> None:
        self.config_manager.add_service(service, config)

    @abstractmethod
    def get_env_var_name(self) -> str:
        """Subclass must implement this method to return the specific environment variable name for the API key."""
        pass

    @abstractmethod
    def create_client(self, api_key: str) -> Any:
        """Subclass must implement this method to create and return the client with the given API key."""
        pass

    def log_error(self, message: str, exc: Exception) -> None:
        logger.error(f"{message}: {exc!s}")

    def log_request_response(self, request: Any, response: Any) -> None:
        """Logs details of the request and response for debugging."""
        logger.debug(f"Request: {request}")
        logger.debug(f"Response: {response}")


class OpenAIBackend(AIBackend):
    def __init__(self, config_manager: ConfigManager, api_key: Optional[str]) -> None:
        super().__init__(config_manager, api_key)

    def get_env_var_name(self) -> str:
        return "OPENAI_API_KEY"

    def create_client(self, api_key: str) -> Client:
        return OpenAI(api_key=api_key)
