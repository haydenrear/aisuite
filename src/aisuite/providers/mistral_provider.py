import os
import typing

from mistralai import Mistral

from aisuite.framework.chat_provider import ChatProvider
from aisuite.framework.tool_utils import SerializedTools


class MistralChatProvider(ChatProvider):
    def __init__(self, **config):
        """
        Initialize the Mistral provider with the given configuration.
        Pass the entire configuration dictionary to the Mistral client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("MISTRAL_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "API key is missing. Please provide it in the config or set the MISTRAL_API_KEY environment variable."
            )
        self.client = Mistral(**config)

    def chat_completions_create(self, model, messages, tools: typing.Optional[SerializedTools]=None, **kwargs):
        return self.client.chat.complete(model=model, messages=messages, **kwargs)
