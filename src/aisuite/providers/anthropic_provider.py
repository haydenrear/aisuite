import typing

import anthropic
from aisuite.framework.chat_provider import ChatProvider
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.tool_utils import SerializedTools

# Define a constant for the default max_tokens value
DEFAULT_MAX_TOKENS = 4096


class AnthropicChatProvider(ChatProvider):
    def __init__(self, **config):
        """
        Initialize the Anthropic provider with the given configuration.
        Pass the entire configuration dictionary to the Anthropic client constructor.
        """

        self.client = anthropic.Anthropic(**config)

    def chat_completions_create(self, model, messages,
                                tools: typing.Optional[SerializedTools] = None,
                                **kwargs):
        # Check if the fist message is a system message
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]
        else:
            system_message = []

        # kwargs.setdefault('max_tokens', DEFAULT_MAX_TOKENS)
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = DEFAULT_MAX_TOKENS

        if tools:
            anthropic_tools = self._convert_to_anthropic_tools(tools)
            kwargs["tools"] = anthropic_tools

        return self.normalize_response(
            self.client.messages.create(
                model=model, system=system_message, messages=messages, **kwargs
            ))

    def normalize_response(self, response):
        """Normalize the response from the Anthropic API to match OpenAI's response format."""
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].message.content = response.content[0].text
        return normalized_response

    def _convert_to_anthropic_tools(self, tools):
        """
        Convert the unified tool format to the format that Anthropic expects.
        """
        if not tools:
            return None

        anthropic_tools = []
        for tool in tools:
            anthropic_tool = {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["args"],  # Assuming 'args' contains the input schema
            }
            anthropic_tools.append(anthropic_tool)
        return anthropic_tools
