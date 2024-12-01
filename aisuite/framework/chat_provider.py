from abc import ABC, abstractmethod

from aisuite.framework.provider_interface import ProviderInterface, Provider


class ChatProvider(Provider, ABC):
    @abstractmethod
    def chat_completions_create(self, model, messages):
        """Abstract method for chat completion calls, to be implemented by each provider."""
        pass

DEFAULT_TEMPERATURE = 0.7

class ChatProviderInterface(ProviderInterface):
    """Defines the expected behavior for provider-specific interfaces."""

    from aisuite.framework.chat_completion_response import ChatCompletionResponse
    def chat_completion_create(self, messages=None, model=None, temperature=DEFAULT_TEMPERATURE) -> ChatCompletionResponse:
        """Create a chat completion using the specified messages, model, and temperature.

        This method must be implemented by subclasses to perform completions.

        Args:
        ----
            messages (list): The chat history.
            model (str): The identifier of the model to be used in the completion.
            temperature (float): The temperature to use in the completion.

        Raises:
        ------
            NotImplementedError: If this method has not been implemented by a subclass.

        """
        raise NotImplementedError(
            "Provider Interface has not implemented chat_completion_create()"
        )
