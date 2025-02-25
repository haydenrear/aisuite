import dataclasses
import typing
from abc import ABC, abstractmethod

import rerankers
import rerankers.results

from aisuite.framework.message import Message
from aisuite.framework.provider_interface import ProviderInterface, Provider


class RerankProvider(Provider, ABC):
    @abstractmethod
    def rerank_create(self, to_embed, model) -> rerankers.reranker.BaseRanker:
        pass

DEFAULT_EMBEDDING_DIM = 1024

@dataclasses.dataclass(init=True)
class RerankRequest:
    query: str
    docs: typing.Union[str, typing.List[str], rerankers.Document, typing.List[rerankers.Document]]
    doc_ids: typing.Optional[typing.Union[typing.List[str], typing.List[int]]] = None


class RerankProviderInterface(ProviderInterface):
    """Defines the expected behavior for provider-specific interfaces."""
    def rerank_create(self, model: str, output_dimensionality=DEFAULT_EMBEDDING_DIM) -> rerankers.reranker.BaseRanker:
        """
        Create an embedding using the specified messages, model, returns of output dimension passed in or default 1024.
        :param to_rank_documents: To create an embedding for.
        :param model: The identifier of the model to be used in the completion.
        :param output_dimensionality:
        :return: embedding of output dimensionality passed in
        :raises NotImplementedError: If this method has not been implemented by a subclass.
        """
        raise NotImplementedError(
            "Provider Interface has not implemented chat_completion_create()"
        )
