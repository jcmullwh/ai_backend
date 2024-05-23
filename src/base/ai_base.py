import os
import logging
import time
from requests.exceptions import RequestException
from retrying import retry
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Dict
from openai import OpenAI

logger = logging.getLogger(__name__)

def retry_if_request_error(exception: Exception) -> bool:
    """Return True if we should retry (in this case when it's a RequestException), False otherwise."""
    return isinstance(exception, RequestException)

class ConfigManager(ABC):
    def __init__(self) -> None:
        # Initialize the configuration dictionary. This dictionary will store configurations for different services.
        self.config: Dict[str, Dict[str, Any]] = {}

        # Setup logging for the ConfigManager. This will help in debugging and logging errors or important information.
        self.logger = logging.getLogger('ConfigManager')
        self.logger.setLevel(logging.DEBUG)  # Set logging level to DEBUG to capture all types of log messages
        handler = logging.StreamHandler()  # Create a stream handler to output logs to the console
        handler.setLevel(logging.DEBUG)  # Set the level for the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Define the log message format
        handler.setFormatter(formatter)  # Set the formatter for the handler
        self.logger.addHandler(handler)  # Add the handler to the logger

    def get_config(self, service: str) -> Dict[str, Any]:
        """
        Retrieve the configuration for a specific service.
        
        Args:
            service (str): The name of the service to retrieve the configuration for.
        
        Returns:
            Dict[str, Any]: The configuration dictionary for the specified service. Returns an empty dictionary if the service is not found.
        """
        return self.config.get(service, {})

    def update_config(self, **kwargs: Dict[str, Any]) -> None:
        """
        Update the configuration for one or more services.
        
        Args:
            kwargs (Dict[str, Any]): Keyword arguments where the key is the service name and the value is the configuration dictionary.
        
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
            else:
                # If the service does not exist in the configuration, add it with the provided values.
                if isinstance(updates, dict):
                    self.config[service] = updates
                elif isinstance(updates, str):
                    # Allow setting a new configuration to a string value directly.
                    self.config[service] = updates
                else:
                    # Log and raise an error if the updates are not of the expected type.
                    error_message = f"Invalid type for new configuration '{service}'. Expected dict or str, got {type(updates).__name__}."
                    self.logger.error(error_message)
                    raise TypeError(error_message)

    def _update_nested_dict(self, target: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """
        Recursively update a nested dictionary with new values.
        
        Args:
            target (Dict[str, Any]): The original dictionary to be updated.
            updates (Dict[str, Any]): The dictionary with new values to update the original dictionary with.
        """
        for key, value in updates.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # If both target and updates have dictionaries for the same key, update the nested dictionary recursively.
                self._update_nested_dict(target[key], value)
            else:
                # Otherwise, simply set the value in the target dictionary.
                target[key] = value

    def set_default(self, service: str, value: Dict[str, Any]) -> None:
        """
        Set the default configuration for a service.
        
        Args:
            service (str): The name of the service.
            value (Dict[str, Any]): The default configuration dictionary for the service.
        
        Raises:
            TypeError: If the provided value is not a dictionary.
        """
        if isinstance(value, dict):
            self.config[service] = value
        else:
            # Log and raise an error if the value is not a dictionary.
            error_message = f"Invalid type for '{service}'. Expected dict, got {type(value).__name__}."
            self.logger.error(error_message)
            raise TypeError(error_message)

    def add_default(self, service: str, value: Dict[str, Any]) -> None:
        """
        Add a new default configuration for a service.
        
        Args:
            service (str): The name of the service.
            value (Dict[str, Any]): The default configuration dictionary for the service.
        
        Raises:
            TypeError: If the provided value is not a dictionary.
            ValueError: If a configuration for the service already exists.
        """
        #if service is none, raise an error
        if service is None:
            error_message = "Service name cannot be None."
            self.logger.error(error_message)
            raise ValueError(error_message)
        
        if value is None:
            error_message = "Value cannot be None."
            self.logger.error(error_message)
            raise ValueError(error_message)
        

        
        if service not in self.config:
            if isinstance(value, dict):
                self.config[service] = value
            else:
                # Log and raise an error if the value is not a dictionary.
                error_message = f"Invalid type for '{service}'. Expected dict, got {type(value).__name__}."
                self.logger.error(error_message)
                raise TypeError(error_message)
        else:
            # Log and raise an error if the service already has a configuration.
            error_message = f"Configuration for '{service}' already exists."
            self.logger.error(error_message)
            raise ValueError(error_message)

    def combine_config(self, service: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Combine the default configuration with the provided configuration.
        
        Args:
            service (str): The name of the service.
            kwargs (Any): Keyword arguments representing the new configuration values to be combined with the default configuration.
        
        Returns:
            Dict[str, Any]: The combined configuration dictionary.
        
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

class AI_Backend(ABC):
    def __init__(self, api_key: Optional[str], config_manager: ConfigManager,**kwargs) -> None:
        if not api_key:
            env_var_name = self.get_env_var_name()
            api_key = os.getenv(env_var_name)
            if not api_key:
                logger.error(f"{env_var_name} is not set.")
                raise ValueError(f"{env_var_name} is not set.")
            
        self.client = self.create_client(api_key)
        self.config_manager = config_manager
    
    @abstractmethod    
    def get_env_var_name(self) -> str:
        """Subclass must implement this method to return the specific environment variable name for the API key."""
        pass

    @abstractmethod
    def create_client(self, api_key: str) -> Any:
        """Subclass must implement this method to create and return the client with the given API key."""
        pass

    def log_error(self, message: str, exc: Exception) -> None:
        logger.error(f"{message}: {str(exc)}")

    @retry(retry_on_exception=retry_if_request_error, stop_max_attempts=3, wait_fixed=2000)
    def safe_api_call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Perform an API call with retry logic for transient errors."""
        try:
            start_time = time.time()
            response = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"API call successful in {duration:.2f} seconds.")
            return response
        except RequestException as e:
            self.log_error("API call failed", e)
            raise
        except Exception as e:
            self.log_error("Unexpected error during API call", e)
            raise

    def log_request_response(self, request: Any, response: Any) -> None:
        """Logs details of the request and response for debugging."""
        logger.debug(f"Request: {request}")
        logger.debug(f"Response: {response}")


class OpenAI_Backend(AI_Backend):
    def __init__(self, api_key: Optional[str], config_manager: ConfigManager) -> None:
        super().__init__(api_key, config_manager)
        
    def get_env_var_name(self) -> str:
        return 'OPENAI_API_KEY'
    
    def create_client(self, api_key: str) -> OpenAI:
        return OpenAI(api_key)
