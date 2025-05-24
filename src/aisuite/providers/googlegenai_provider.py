"""The interface to Google's Vertex AI."""
import os
import typing

import google.generativeai as genai
from google.generativeai.types import text_types, ToolsType
from langchain_core.tools import BaseTool

from aisuite.framework.chat_provider import DEFAULT_TEMPERATURE, ChatProvider
from aisuite.framework.embedding_provider import EmbeddingProviderInterface, DEFAULT_EMBEDDING_DIM
from aisuite.framework.tool_utils import SerializedTools
from aisuite.providers.google_provider_shared import transform_roles, normalize_response, convert_openai_to_google_ai


class GoogleGenAiProvider:
    """Implements the ProviderInterface for interacting with Google's GenAi API."""

    def __init__(self, **config):
        """Set up the Google AI client with a project ID."""
        api_key = config.get("api_key") or os.getenv("GOOGLE_GEN_AI_API_KEY")
        if not api_key:
            raise EnvironmentError("API Key not provided for google Gen AI.")

        genai.configure(api_key=api_key)


class GooglegenaiChatProvider(GoogleGenAiProvider, ChatProvider):

    # TODO: could this return a function with closure containing the chat instead?
    def chat_completions_create(self, model, messages,
                                tools: typing.Optional[SerializedTools] = None, **kwargs):
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

        # Set the temperature if provided, otherwise use the default
        temperature = kwargs.get("temperature", DEFAULT_TEMPERATURE)

        # Transform the roles in the messages
        transformed_messages = transform_roles(messages)

        # Convert the messages to the format expected Google
        final_message_history = convert_openai_to_google_ai(
            transformed_messages[:-1])

        # Get the last message from the transformed messages
        last_message = transformed_messages[-1]["content"]

        tool_call_converted = self.convert_to_tools_types(tools)

        # Create the GenerativeModel with the specified model and generation configuration
        model = genai.GenerativeModel(
            model, generation_config=genai.GenerationConfig(temperature=temperature),
            tools=tool_call_converted
        )

        # Start a chat with the GenerativeModel and send the last message
        chat = model.start_chat(history=final_message_history)
        response = chat.send_message(last_message)

        # Convert the response to the format expected by the OpenAI API
        return normalize_response(response)

    def convert_to_tools_types(self, tools) -> typing.Optional[ToolsType]:
        if not tools:
            return None

        tool_types = []
        for tool in tools:
            tool_types.append(
                {
                    "function_declarations": [
                        {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": {
                                "type": "OBJECT",
                                "properties": tool["args"],
                            },
                        }
                    ]
                }
            )
        return tool_types

class GooglegenaiEmbeddingProvider(GoogleGenAiProvider, EmbeddingProviderInterface):

    """Defines the expected behavior for provider-specific interfaces."""
    def embedding_create(self, to_embed, model, **kwargs) \
            -> text_types.EmbeddingDict | text_types.BatchEmbeddingDict:
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
        return genai.embed_content(model=model, content=to_embed,
                                   output_dimensionality=kwargs.get("output_dimensionality", DEFAULT_EMBEDDING_DIM))

