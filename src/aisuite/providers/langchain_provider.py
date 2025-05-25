import os
import typing
import json

import httpx
from aisuite.framework.chat_provider import ChatProvider, DEFAULT_TEMPERATURE
from aisuite.provider import LLMError
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.tool_utils import SerializedTools
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain.tools import Tool, StructuredTool
from langchain.callbacks.base import BaseCallbackHandler


class LangchainChatProvider(ChatProvider):
    """
    Langchain Provider that supports various LLM backends through the langchain interface.
    This provider implements chat completions through langchain's chat models.
    """

    def __init__(self, **config):
        """
        Initialize the provider with the given configuration.
        The token is fetched from the config or environment variables.
        """
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("API Key not provided for Langchain provider.")
        
        # Other optional configurations
        self.base_url = config.get("base_url")
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)
        
        # Pre-load model if specified in config
        self.default_model = config.get("model")
        self.model_instances = {}  # Cache for model instances

    def chat_completions_create(self, model, messages, tools: typing.Optional[SerializedTools] = None, **kwargs):
        """
        Makes a request to the Inference API endpoint using Langchain's ChatOpenAI.
        
        Args:
            model (str): The model identifier to use.
            messages (list): A list of message objects with role and content.
            tools (Optional[SerializedTools]): Tools to be used by the model.
            **kwargs: Additional parameters for the API call.
        
        Returns:
            ChatCompletionResponse: A normalized response object.
        """
        # Set temperature if provided, otherwise use default
        temperature = kwargs.get("temperature", DEFAULT_TEMPERATURE)
        
        # Use provided model or default model from config
        model_name = model or self.default_model
        if not model_name:
            raise LLMError("No model specified for Langchain provider")
        
        # Get or create the model instance
        if model_name not in self.model_instances:
            # Create the ChatOpenAI instance with the specified model and configuration
            chat_model = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=self.api_key,
                max_retries=self.max_retries,
                request_timeout=self.timeout
            )
            
            if self.base_url:
                chat_model.openai_api_base = self.base_url
                
            self.model_instances[model_name] = chat_model
        else:
            chat_model = self.model_instances[model_name]
            # Update temperature if different from cached instance
            if chat_model.temperature != temperature:
                chat_model.temperature = temperature
        
        # Convert messages to Langchain's format
        langchain_messages = []
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")
            
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                # Handle function call in assistant messages
                if "function_call" in message:
                    function_call = message["function_call"]
                    msg = AIMessage(content=content)
                    msg.additional_kwargs["function_call"] = function_call
                    langchain_messages.append(msg)
                else:
                    langchain_messages.append(AIMessage(content=content))
            elif role == "function":
                langchain_messages.append(FunctionMessage(
                    name=message.get("name", ""),
                    content=content
                ))
        
        # Handle tools if provided
        if tools:
            langchain_tools = []
            for tool in tools:
                # Create a more structured tool definition
                tool_func = lambda **kwargs: json.dumps(kwargs)  # Placeholder function
                
                # Extract parameters schema from the tool definition
                parameters_schema = {}
                if "args" in tool:
                    parameters_schema = {
                        "type": "object",
                        "properties": tool["args"],
                        "required": [k for k, v in tool["args"].items() if v.get("required", False)]
                    }
                
                structured_tool = StructuredTool(
                    name=tool["name"],
                    description=tool["description"],
                    func=tool_func,
                    args_schema=parameters_schema
                )
                langchain_tools.append(structured_tool)
            
            # Set the functions on the model
            chat_model.functions = [t.to_openai_function() for t in langchain_tools]
        
        # Make the request
        try:
            response = chat_model.generate([langchain_messages])
            return self._normalize_response(response)
        except Exception as e:
            raise LLMError(f"Error in Langchain chat completion: {str(e)}")

    def _normalize_response(self, response_data):
        """
        Normalize the response to a common format (ChatCompletionResponse).
        
        Args:
            response_data: The raw response from Langchain.
            
        Returns:
            ChatCompletionResponse: A normalized response object.
        """
        try:
            # Extract the generation from the response
            generation = response_data.generations[0][0]
            message = generation.message
            
            # Initialize response components
            function_call = None
            tool_calls = None
            
            # Check for function call or tool calls in additional kwargs
            if hasattr(message, "additional_kwargs"):
                additional_kwargs = message.additional_kwargs
                
                # Handle function_call (legacy format)
                if "function_call" in additional_kwargs:
                    function_call = additional_kwargs["function_call"]
                
                # Handle tool_calls (newer format)
                if "tool_calls" in additional_kwargs:
                    tool_calls = additional_kwargs["tool_calls"]
            
            # Build the message component
            assistant_message = {
                "role": "assistant",
                "content": message.content,
            }
            
            # Add function_call or tool_calls if present
            if function_call:
                assistant_message["function_call"] = function_call
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            
            # Create a normalized response
            normalized_response = ChatCompletionResponse(
                id=response_data.llm_output.get("id", ""),
                object="chat.completion",
                created=response_data.llm_output.get("created", 0),
                model=response_data.llm_output.get("model", ""),
                choices=[
                    {
                        "index": 0,
                        "message": assistant_message,
                        "finish_reason": generation.finish_reason or "stop"
                    }
                ],
                usage=response_data.llm_output.get("usage", {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                })
            )
            
            return normalized_response
        except Exception as e:
            raise LLMError(f"Error normalizing Langchain response: {str(e)}")
