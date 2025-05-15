import os
import typing

import httpx
from aisuite.framework.chat_provider import ChatProvider
from aisuite.provider import LLMError
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.tool_utils import SerializedTools


class LangchainChatProvider(ChatProvider):
    """
    HuggingFace Provider using httpx for direct API calls.
    Currently, this provider support calls to HF serverless Inference Endpoints
    which uses Text Generation Inference (TGI) as the backend.
    TGI is OpenAI protocol compliant.
    https://huggingface.co/inference-endpoints/
    """

    def __init__(self, **config):
        """
        Initialize the provider with the given configuration.
        The token is fetched from the config or environment variables.
        """
        # TODO
        pass

    def chat_completions_create(self, model, messages, tools: typing.Optional[SerializedTools] = None, **kwargs):
        """
        Makes a request to the Inference API endpoint using httpx.
        """
        # TODO:
        pass

    def _normalize_response(self, response_data):
        """
        Normalize the response to a common format (ChatCompletionResponse).
        """
        # TODO:
        pass
