from unittest.mock import MagicMock, patch
import json
import pytest

import httpx
from aisuite.providers.fireworks_provider import FireworksChatProvider
from aisuite.provider import LLMError


@pytest.fixture(autouse=True)
def set_fireworks_env_vars(monkeypatch):
    """Fixture to set environment variables for Fireworks tests."""
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-api-key")


def test_fireworks_provider_init():
    """Test initialization of the Fireworks provider."""
    # Test with API key from environment
    provider = FireworksChatProvider()
    assert provider.api_key == "test-api-key"
    assert provider.timeout == 30
    
    # Test with explicit API key
    provider = FireworksChatProvider(api_key="explicit-api-key", timeout=60)
    assert provider.api_key == "explicit-api-key"
    assert provider.timeout == 60
    
    # Test missing API key
    with pytest.raises(ValueError, match="Fireworks API key is missing"):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        FireworksChatProvider()


def test_fireworks_chat_completions_basic():
    """Test basic chat completions without tools."""
    provider = FireworksChatProvider()
    
    # Create mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "llama-v2-7b",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                },
                "finish_reason": "stop",
                "index": 0
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()
    
    # Mock the httpx post method
    with patch('httpx.post', return_value=mock_response) as mock_post:
        response = provider.chat_completions_create(
            model="llama-v2-7b",
            messages=[{"role": "user", "content": "Hello!"}],
            temperature=0.7
        )
        
        # Verify the post method was called with the correct arguments
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == provider.BASE_URL
        assert call_args[1]['headers']['Authorization'] == "Bearer test-api-key"
        assert call_args[1]['json']['model'] == "llama-v2-7b"
        assert call_args[1]['json']['messages'][0]['content'] == "Hello!"
        assert call_args[1]['json']['temperature'] == 0.7
    
    # Verify the response was normalized correctly
    assert response.choices[0].message.content == "Hello! How can I help you today?"


def test_fireworks_chat_completions_with_tools():
    """Test chat completions with tools."""
    provider = FireworksChatProvider()
    
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
    
    # Create mock response with tool calls
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "llama-v2-7b",
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
        ]
    }
    mock_response.raise_for_status = MagicMock()
    
    # Mock the httpx post method
    with patch('httpx.post', return_value=mock_response) as mock_post:
        response = provider.chat_completions_create(
            model="llama-v2-7b",
            messages=[{"role": "user", "content": "What's the weather in New York?"}],
            tools=tools
        )
        
        # Verify the tools were included in the request
        call_args = mock_post.call_args
        assert "tools" in call_args[1]['json']
        assert call_args[1]['json']['tools'] == tools
    
    # Verify the response was normalized correctly
    assert response.choices[0].message.content == "I'll check the weather for you."
    assert response.choices[0].message.tool_calls is not None
    assert len(response.choices[0].message.tool_calls) == 1
    assert response.choices[0].message.tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(response.choices[0].message.tool_calls[0]["function"]["arguments"])["location"] == "New York"
    assert response.choices[0].finish_reason == "tool_calls"


def test_fireworks_chat_completions_with_function_call():
    """Test chat completions with function calls (legacy format)."""
    provider = FireworksChatProvider()
    
    # Create mock response with function call
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "llama-v2-7b",
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
    mock_response.raise_for_status = MagicMock()
    
    # Mock the httpx post method
    with patch('httpx.post', return_value=mock_response):
        response = provider.chat_completions_create(
            model="llama-v2-7b",
            messages=[{"role": "user", "content": "Tell me about climate change"}]
        )
    
    # Verify the response was normalized correctly
    assert response.choices[0].message.content == "I'll search for information about climate change."
    assert response.choices[0].message.function_call is not None
    assert response.choices[0].message.function_call["name"] == "search"
    assert json.loads(response.choices[0].message.function_call["arguments"])["query"] == "climate change impact"
    assert response.choices[0].finish_reason == "function_call"


def test_fireworks_error_handling():
    """Test error handling in the Fireworks provider."""
    provider = FireworksChatProvider()
    
    # Test HTTP status error
    with patch('httpx.post', side_effect=httpx.HTTPStatusError("Error", request=MagicMock(), response=MagicMock())):
        with pytest.raises(LLMError) as excinfo:
            provider.chat_completions_create(
                model="llama-v2-7b",
                messages=[{"role": "user", "content": "Hello"}]
            )
        assert "Fireworks AI request failed" in str(excinfo.value)
    
    # Test general exception
    with patch('httpx.post', side_effect=Exception("General error")):
        with pytest.raises(LLMError) as excinfo:
            provider.chat_completions_create(
                model="llama-v2-7b",
                messages=[{"role": "user", "content": "Hello"}]
            )
        assert "An error occurred" in str(excinfo.value)


def test_normalize_response():
    """Test response normalization."""
    provider = FireworksChatProvider()
    
    # Test basic response
    basic_response = {
        "choices": [
            {
                "message": {
                    "content": "Simple response",
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }
        ]
    }
    norm_response = provider._normalize_response(basic_response)
    assert norm_response.choices[0].message.content == "Simple response"
    assert norm_response.choices[0].finish_reason == "stop"
    
    # Test response with tool calls
    tool_response = {
        "choices": [
            {
                "message": {
                    "content": "Tool response",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "test_tool",
                                "arguments": json.dumps({"arg1": "value1"})
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ]
    }
    norm_response = provider._normalize_response(tool_response)
    assert norm_response.choices[0].message.content == "Tool response"
    assert norm_response.choices[0].message.tool_calls is not None
    assert norm_response.choices[0].finish_reason == "tool_calls"
    
    # Test response with function call
    function_response = {
        "choices": [
            {
                "message": {
                    "content": "Function response",
                    "role": "assistant",
                    "function_call": {
                        "name": "test_function",
                        "arguments": json.dumps({"arg1": "value1"})
                    }
                },
                "finish_reason": "function_call"
            }
        ]
    }
    norm_response = provider._normalize_response(function_response)
    assert norm_response.choices[0].message.content == "Function response"
    assert norm_response.choices[0].message.function_call is not None
    assert norm_response.choices[0].finish_reason == "function_call"