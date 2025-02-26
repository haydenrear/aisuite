import abc
import dataclasses
import typing
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import rerankers.results

from aisuite.framework.provider_interface import ProviderInterface, Provider


class AiSuiteReranker(rerankers.reranker.BaseRanker, abc.ABC):

    @abc.abstractmethod
    def do_rank(self, data: dict[str, ...]) -> rerankers.results.RankedResults:
        pass

    @abc.abstractmethod
    def __call__(self,data: dict[str, ...]) -> rerankers.results.RankedResults:
        pass

class RerankProvider(Provider, ABC):
    @abstractmethod
    def rerank_create(self, to_embed, model) -> AiSuiteReranker:
        pass

DEFAULT_EMBEDDING_DIM = 1024

class RerankProviderInterface(ProviderInterface):
    """Defines the expected behavior for provider-specific interfaces."""
    def rerank_create(self, model: str, output_dimensionality=DEFAULT_EMBEDDING_DIM) -> AiSuiteReranker:
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
