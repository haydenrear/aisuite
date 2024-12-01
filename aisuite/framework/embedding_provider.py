from abc import ABC, abstractmethod

from aisuite.framework.message import Message
from aisuite.framework.provider_interface import ProviderInterface, Provider


class EmbeddingProvider(Provider, ABC):
    @abstractmethod
    def embedding_create(self, to_embed, model):
        pass

DEFAULT_EMBEDDING_DIM = 1024

class EmbeddingProviderInterface(ProviderInterface):
    """Defines the expected behavior for provider-specific interfaces."""
    def embedding_create(self, to_embed, model, output_dimensionality=DEFAULT_EMBEDDING_DIM) -> None:
        """
        Create an embedding using the specified messages, model, returns of output dimension passed in or default 1024.
        :param to_embed: To create an embedding for.
        :param model: The identifier of the model to be used in the completion.
        :param output_dimensionality:
        :return: embedding of output dimensionality passed in
        :raises NotImplementedError: If this method has not been implemented by a subclass.
        """
        raise NotImplementedError(
            "Provider Interface has not implemented chat_completion_create()"
        )
