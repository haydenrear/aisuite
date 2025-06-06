import typing
import urllib.request
import json
import os

from aisuite.framework.chat_provider import ChatProvider
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.tool_utils import SerializedTools


class AzureChatProvider(ChatProvider):
    def __init__(self, **config):
        self.base_url = config.get("base_url") or os.getenv("AZURE_BASE_URL")
        self.api_key = config.get("api_key") or os.getenv("AZURE_API_KEY")
        if not self.api_key:
            raise ValueError("For Azure, api_key is required.")
        if not self.base_url:
            raise ValueError(
                "For Azure, base_url is required. Check your deployment page for a URL like this - https://<model-deployment-name>.<region>.models.ai.azure.com"
            )

    def chat_completions_create(self, model, messages, tools: typing.Optional[SerializedTools]=None, **kwargs):
        url = f"https://{model}.westus3.models.ai.azure.com/v1/chat/completions"
        url = f"https://{self.base_url}/chat/completions"
        if self.base_url:
            url = f"{self.base_url}/chat/completions"

        # Remove 'stream' from kwargs if present
        kwargs.pop("stream", None)
        data = {"messages": messages, **kwargs}
        
        # Add tools to the request if provided
        if tools:
            data["tools"] = tools

        body = json.dumps(data).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": self.api_key}

        try:
            req = urllib.request.Request(url, body, headers)
            with urllib.request.urlopen(req) as response:
                result = response.read()
                resp_json = json.loads(result)
                completion_response = ChatCompletionResponse()
                # Extract basic message content
                completion_response.choices[0].message.content = resp_json["choices"][0]["message"]["content"]
                
                # Check for and extract tool calls if present
                if "tool_calls" in resp_json["choices"][0]["message"]:
                    completion_response.choices[0].message.tool_calls = resp_json["choices"][0]["message"]["tool_calls"]
                    completion_response.choices[0].finish_reason = "tool_calls"
                
                # Check for and extract function call if present (legacy format)
                elif "function_call" in resp_json["choices"][0]["message"]:
                    completion_response.choices[0].message.function_call = resp_json["choices"][0]["message"]["function_call"]
                    completion_response.choices[0].finish_reason = "function_call"
                
                return completion_response

        except urllib.error.HTTPError as error:
            error_message = f"The request failed with status code: {error.code}\n"
            error_message += f"Headers: {error.info()}\n"
            error_message += error.read().decode("utf-8", "ignore")
            raise Exception(error_message)
