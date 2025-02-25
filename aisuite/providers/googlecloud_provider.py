"""The interface to Google's Vertex AI."""

import os
from typing import List, Optional, Union

import google.generativeai as cloud
import rerankers.results
from google.cloud import discoveryengine_v1 as discoveryengine
from rerankers import Document
from rerankers.utils import (
    prep_docs,
)

from aisuite.framework.chat_provider import ChatProvider
from aisuite.framework.embedding_provider import DEFAULT_EMBEDDING_DIM
from aisuite.framework.rerank_provider import RerankProviderInterface
from aisuite.framework.rerank_util import create_ranking_records
from aisuite.providers.googlegenai_provider import GoogleGenAiProvider

GOOGLE_APP_CRED_KEY = "GOOGLE_APPLICATION_CREDENTIALS"


class GoogleCloudProvider:
    """Implements the ProviderInterface for interacting with Google's Cloud API."""

    def __init__(self, **config):
        """Set up the Google AI client with a project ID."""
        api_key = config.get("api_key") or os.getenv("GOOGLE_GEN_AI_API_KEY")
        if not api_key:
            raise EnvironmentError("API Key not provided for google Gen AI.")

        cloud.configure(api_key=api_key)


class GooglecloudChatProvider(GoogleCloudProvider, ChatProvider):

    # TODO: could this return a function with closure containing the chat instead?
    def chat_completions_create(self, model, messages, **kwargs):
        """Request chat completions from the Google AI API.

        Args:
        ----
            model (str): Identifies the specific provider/model to use.
            messages (list of dict): A list of message objects in chat history.
            kwargs (dict): Optional arguments for the Google AI API.

        Returns:
        -------
            The ChatCompletionResponse with the completion result.

        """

        raise NotImplementedError("Chat provider not implemented for google cloud.")


class GooglecloudReranker(rerankers.reranker.BaseRanker):

    def __init__(self, model: str, project_id: str, application_credential: str, **kwargs):
        if GOOGLE_APP_CRED_KEY not in os.environ.keys():
            os.environ[GOOGLE_APP_CRED_KEY] = application_credential

        client = discoveryengine.RankServiceClient()

        self.model = model


        self.ranking_config = client.ranking_config_path(
            project=project_id,
            location="global",
            ranking_config="default_ranking_config",
        )

        self.gen_ai = self.do_rank

    def do_rank(self,
                query: str,
                docs: Union[str, List[str], rerankers.results.Document, List[rerankers.results.Document]],
                doc_ids: Optional[Union[List[str], List[int]]] = None,
                metadata: Optional[list[dict]] = None):
        return discoveryengine.RankRequest(
            ranking_config=self.ranking_config,
            model=self.model,
            top_n=len(docs) + 1,
            query=query,
            records=create_ranking_records(docs, doc_ids, metadata))

    def __call__(self,
                 query: str,
                 docs: Union[str, List[str], rerankers.results.Document, List[rerankers.results.Document]],
                 doc_ids: Optional[Union[List[str], List[int]]] = None,
                 metadata: Optional[List[dict]] = None):
        gen_ai_rerank_call = self.gen_ai(query, docs, doc_ids)
        return gen_ai_rerank_call


    def rank(
            self,
            query: str,
            docs: Union[str, List[str], rerankers.results.Document, List[rerankers.results.Document]],
            doc_ids: Optional[Union[List[str], List[int]]] = None,
            metadata: Optional[List[dict]] = None
    ) -> rerankers.results.RankedResults:
        """
        Ranks a list of documents based on their relevance to the query.
        """
        docs = prep_docs(docs, doc_ids, metadata)
        scores = self.gen_ai(query, docs, doc_ids, metadata)
        reranked = self._parse_ranked_results(query, scores)
        return reranked

    @staticmethod
    def _parse_ranked_results(query, scores):
        reranked = rerankers.results.RankedResults([
            rerankers.results.Result(Document(next_doc.content), None, i)
            for i, next_doc in enumerate(scores.records)
        ],
            query,
            False)
        return reranked

    def score(self, query: str, doc: str) -> float:
        raise NotImplementedError("Did not implement score function for API call.")

class GooglecloudRerankProvider(GoogleCloudProvider, RerankProviderInterface):

    def __init__(self, **config):
        """Set up the Google AI client with a project ID."""
        self.api_key = config.get("api_key") or os.getenv("GOOGLE_GEN_AI_API_KEY")
        self.project_id = config.get("project_id") or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        self.application_credential = config.get("application_credential") or os.getenv("GOOGLE_CLOUD_APPLICATION_CREDENTIAL")


    """Defines the expected behavior for provider-specific interfaces."""
    def rerank_create(self, model: str, output_dimensionality=DEFAULT_EMBEDDING_DIM, **kwargs) -> rerankers.reranker.BaseRanker:
        """Create an embedding using the specified messages, model, and temperature.

        This method must be implemented by subclasses to perform completions.

        Args:
        ----
            messages: To create an embedding for.
            model (str): The identifier of the model to be used in the completion.
            temperature (float): The temperature to use in the completion.

        Raises:
        ------
            NotImplementedError: If this method has not been implemented by a subclass.

        """
        return GooglecloudReranker(model, self.project_id, self.application_credential, **kwargs)

