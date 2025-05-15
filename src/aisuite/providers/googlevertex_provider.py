"""The interface to Google's Vertex AI."""

import os
import typing

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from aisuite.framework import ChatProviderInterface
from aisuite.framework.tool_utils import SerializedTools
from aisuite.providers.google_provider_shared import transform_roles, normalize_response, convert_openai_to_google_ai
from aisuite.framework.chat_provider import DEFAULT_TEMPERATURE


class GoogleProvider(ChatProviderInterface):
    """Implements the ProviderInterface for interacting with Google's Vertex AI."""

    def __init__(self, **config):
        """Set up the Google AI client with a project ID."""
        self.project_id = config.get("project_id") or os.getenv("GOOGLE_PROJECT_ID")
        self.location = config.get("region") or os.getenv("GOOGLE_REGION")
        self.app_creds_path = config.get("application_credentials") or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )

        if not self.project_id or not self.location or not self.app_creds_path:
            raise EnvironmentError(
                "Missing one or more required Google environment variables: "
                "GOOGLE_PROJECT_ID, GOOGLE_REGION, GOOGLE_APPLICATION_CREDENTIALS. "
                "Please refer to the setup guide: /guides/google.md."
            )

        vertexai.init(project=self.project_id, location=self.location, credentials=self.app_creds_path)


class GooglevertexChatProvider(GoogleProvider):

    # TODO: could this return a function with closure containing the chat instead?
    def chat_completions_create(self, model, messages, tools: typing.Optional[SerializedTools]=None, **kwargs):
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
            transformed_messages[:-1]
        )

        # Get the last message from the transformed messages
        last_message = transformed_messages[-1]["content"]

        # Create the GenerativeModel with the specified model and generation configuration
        model = GenerativeModel(
            model, generation_config=GenerationConfig(temperature=temperature)
        )

        # Start a chat with the GenerativeModel and send the last message
        chat = model.start_chat(history=final_message_history)
        response = chat.send_message(last_message)

        # Convert the response to the format expected by the OpenAI API
        return normalize_response(response)

