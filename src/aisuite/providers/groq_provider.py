import os
import typing

import groq

from aisuite.framework.chat_provider import ChatProvider
from aisuite.framework.tool_utils import SerializedTools


class GroqChatProvider(ChatProvider):
    def __init__(self, **config):
        """
        Initialize the Groq provider with the given configuration.
        Pass the entire configuration dictionary to the Groq client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("GROQ_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                " API key is missing. Please provide it in the config or set the GROQ_API_KEY environment variable."
            )
        self.client = groq.Groq(**config)

    def chat_completions_create(self, model, messages, tools: typing.Optional[SerializedTools]=None, **kwargs):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs  # Pass any additional arguments to the Groq API
        )
