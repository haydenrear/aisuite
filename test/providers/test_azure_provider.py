from unittest.mock import MagicMock, patch
import json
import pytest
import urllib.request
from urllib.error import HTTPError

from aisuite.providers.azure_provider import AzureChatProvider


@pytest.fixture(autouse=True)
def set_azure_env_vars(monkeypatch):
    """Fixture to set environment variables for Azure tests."""
    monkeypatch.setenv("AZURE_API_KEY", "test-api-key")
    monkeypatch.setenv("AZURE_BASE_URL", "test-deployment.westus3.models.ai.azure.com")


def test_azure_provider_init():
    """Test the initialization of the Azure provider."""
    # Test with environment variables
    provider = AzureChatProvider()
    assert provider.api_key == "test-api-key"
    assert provider.base_url == "test-deployment.westus3.models.ai.azure.com"
    
    # Test with explicit config
    provider = AzureChatProvider(
        api_key="explicit-api-key",
        base_url="custom-deployment.eastus.models.ai.azure.com"
    )
    assert provider.api_key == "explicit-api-key"
    assert provider.base_url == "custom-deployment.eastus.models.ai.azure.com"
    
    # Test missing API key
    with pytest.raises(ValueError, match="api_key is required"):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.delenv("AZURE_API_KEY", raising=False)
        AzureChatProvider(base_url="test.com")
    
    # Test missing base URL
    with pytest.raises(ValueError, match="base_url is required"):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.delenv("AZURE_BASE_URL", raising=False)
        AzureChatProvider(api_key="test-key")


class MockResponse:
    """Mock class for urllib.request.urlopen response."""
    
    def __init__(self, json_data, status=200):
        self.json_data = json_data
        self.status = status
    
    def read(self):
        return json.dumps(self.json_data).encode('utf-8')
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


def test_azure_chat_completions_basic():
    """Test basic chat completions without tools."""
    provider = AzureChatProvider()
    
    # Mock response for a basic chat completion
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                },
                "finish_reason": "stop",
                "index": 0
            }
        ],
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20
        }
    }
    
    with patch('urllib.request.urlopen', return_value=MockResponse(mock_response)):
        with patch('urllib.request.Request', return_value=MagicMock()):
            response = provider.chat_completions_create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello!"}],
                temperature=0.7
            )
    
    # Verify the response was normalized correctly
    assert response.choices[0].message.content == "Hello! How can I help you today!"


def test_azure_chat_completions_with_tools():
    """Test chat completions with tools."""
    provider = AzureChatProvider()
    
    # Define tools
    tools = [
        {
            "name": "get_weather", 
            "description": "Get the current weather in a location",
            "args": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}
            }
        }
    ]
    
    # Mock response with tool calls
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I'll check the weather for you.",
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"location": "New York"})
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls",
                "index": 0
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 15,
            "total_tokens": 35
        }
    }
    
    # Mock the request
    with patch('urllib.request.urlopen', return_value=MockResponse(mock_response)):
        with patch('urllib.request.Request') as mock_request:
            response = provider.chat_completions_create(
                model="gpt-4",
                messages=[{"role": "user", "content": "What's the weather in New York?"}],
                tools=tools
            )
            
            # Verify tools were included in the request
            request_args = mock_request.call_args
            request_body = json.loads(request_args[0][1].decode('utf-8'))
            assert "tools" in request_body
            assert request_body["tools"] == tools
    
    # Verify the response was normalized correctly
    assert response.choices[0].message.content == "I'll check the weather for you."
    assert response.choices[0].message.tool_calls is not None
    assert len(response.choices[0].message.tool_calls) == 1
    assert response.choices[0].message.tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(response.choices[0].message.tool_calls[0]["function"]["arguments"])["location"] == "New York"
    assert response.choices[0].finish_reason == "tool_calls"


def test_azure_chat_completions_with_function_call():
    """Test chat completions with function calls (legacy format)."""
    provider = AzureChatProvider()
    
    # Mock response with function call
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I'll search for information about climate change.",
                    "function_call": {
                        "name": "search",
                        "arguments": json.dumps({"query": "climate change impact"})
                    }
                },
                "finish_reason": "function_call",
                "index": 0
            }
        ]
    }
    
    with patch('urllib.request.urlopen', return_value=MockResponse(mock_response)):
        with patch('urllib.request.Request', return_value=MagicMock()):
            response = provider.chat_completions_create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Tell me about climate change"}]
            )
    
    # Verify the response was normalized correctly
    assert response.choices[0].message.content == "I'll search for information about climate change."
    assert response.choices[0].message.function_call is not None
    assert response.choices[0].message.function_call["name"] == "search"
    assert json.loads(response.choices[0].message.function_call["arguments"])["query"] == "climate change impact"
    assert response.choices[0].finish_reason == "function_call"


def test_azure_error_handling():
    """Test error handling in the Azure provider."""
    provider = AzureChatProvider()
    
    # Mock an HTTP error
    http_error = HTTPError(
        url="https://test-deployment.westus3.models.ai.azure.com/chat/completions",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=None
    )
    
    with patch('urllib.request.urlopen', side_effect=http_error):
        with patch('urllib.request.Request', return_value=MagicMock()):
            with pytest.raises(Exception) as excinfo:
                provider.chat_completions_create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hello"}]
                )
            
            assert "The request failed with status code: 401" in str(excinfo.value)