import os
import typing

import httpx
from aisuite.framework.chat_provider import ChatProvider
from aisuite.framework.tool_utils import SerializedTools
from aisuite.provider import LLMError
from aisuite.framework import ChatCompletionResponse


class FireworksChatProvider(ChatProvider):
    """
    Fireworks AI Provider using httpx for direct API calls.
    """

    BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

    def __init__(self, **config):
        """
        Initialize the Fireworks provider with the given configuration.
        The API key is fetched from the config or environment variables.
        """
        self.api_key = config.get("api_key", os.getenv("FIREWORKS_API_KEY"))
        if not self.api_key:
            raise ValueError(
                "Fireworks API key is missing. Please provide it in the config or set the FIREWORKS_API_KEY environment variable."
            )

        # Optionally set a custom timeout (default to 30s)
        self.timeout = config.get("timeout", 30)

    def chat_completions_create(self, model, messages, tools: typing.Optional[SerializedTools]=None, **kwargs):
        """
        Makes a request to the Fireworks AI chat completions endpoint using httpx.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "messages": messages,
            **kwargs,  # Pass any additional arguments to the API
        }
        
        # Add tools to the request if provided
        if tools:
            data["tools"] = tools

        try:
            # Make the request to Fireworks AI endpoint.
            response = httpx.post(
                self.BASE_URL, json=data, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as http_err:
            raise LLMError(f"Fireworks AI request failed: {http_err}")
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

        # Return the normalized response
        return self._normalize_response(response.json())

    def _normalize_response(self, response_data):
        """
        Normalize the response to a common format (ChatCompletionResponse).
        """
        normalized_response = ChatCompletionResponse()
        response_choice = response_data["choices"][0]
        response_message = response_choice["message"]
        
        # Extract content
        normalized_response.choices[0].message.content = response_message["content"]
        
        # Extract tool_calls if present
        if "tool_calls" in response_message:
            normalized_response.choices[0].message.tool_calls = response_message["tool_calls"]
            normalized_response.choices[0].finish_reason = "tool_calls"
        
        # Extract function_call if present (legacy format)
        elif "function_call" in response_message:
            normalized_response.choices[0].message.function_call = response_message["function_call"]
            normalized_response.choices[0].finish_reason = "function_call"
        else:
            normalized_response.choices[0].finish_reason = response_choice.get("finish_reason", "stop")
            
        return normalized_response
