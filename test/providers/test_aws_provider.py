from unittest.mock import MagicMock, patch

import pytest
import json
import boto3

from aisuite.providers.aws_provider import AwsChatProvider


@pytest.fixture(autouse=True)
def set_aws_env_vars(monkeypatch):
    """Fixture to set environment variables for AWS tests."""
    monkeypatch.setenv("AWS_REGION_NAME", "us-west-2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")


def test_aws_provider_init():
    """Test the initialization of the AWS provider."""
    provider = AwsChatProvider()
    assert provider.region_name == "us-west-2"
    
    # Test custom region
    provider = AwsChatProvider(region_name="us-east-1")
    assert provider.region_name == "us-east-1"


def test_aws_chat_completions_basic():
    """Test basic chat completions without tools."""
    provider = AwsChatProvider()
    
    # Mock the boto3 client
    mock_client = MagicMock()
    provider.client = mock_client
    
    # Mock the response from AWS Bedrock
    mock_response = {
        "output": {
            "message": {
                "content": [{"text": "Hello! How can I help you today?"}]
            }
        }
    }
    mock_client.converse.return_value = mock_response
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"}
    ]
    
    response = provider.chat_completions_create(
        model="anthropic.claude-v2",
        messages=messages,
        temperature=0.7
    )
    
    # Verify the response was normalized correctly
    assert response.choices[0].message.content == "Hello! How can I help you today?"
    
    # Verify the Bedrock API was called with the correct parameters
    mock_client.converse.assert_called_once()
    call_args = mock_client.converse.call_args[1]
    assert call_args["modelId"] == "anthropic.claude-v2"
    assert len(call_args["messages"]) == 1  # System message moved to system parameter
    assert call_args["messages"][0]["role"] == "user"
    assert call_args["system"] == [{"text": "You are a helpful assistant"}]
    assert call_args["inferenceConfig"]["temperature"] == 0.7


def test_aws_chat_completions_with_tools():
    """Test chat completions with tools."""
    provider = AwsChatProvider()
    
    # Mock the boto3 client
    mock_client = MagicMock()
    provider.client = mock_client
    
    # Mock the response from AWS Bedrock
    mock_response = {
        "output": {
            "message": {
                "content": [{"text": "I'll check the weather for you."}],
                "toolUse": [
                    {
                        "toolName": "get_weather",
                        "input": json.dumps({"location": "New York"})
                    }
                ]
            }
        }
    }
    mock_client.converse.return_value = mock_response
    
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
    
    messages = [
        {"role": "user", "content": "What's the weather in New York?"}
    ]
    
    response = provider.chat_completions_create(
        model="anthropic.claude-v2",
        messages=messages,
        tools=tools
    )
    
    # Verify the response was normalized correctly
    assert response.choices[0].message.content == "I'll check the weather for you."
    assert response.choices[0].message.tool_calls is not None
    assert len(response.choices[0].message.tool_calls) == 1
    assert response.choices[0].message.tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(response.choices[0].message.tool_calls[0]["function"]["arguments"])["location"] == "New York"
    assert response.choices[0].finish_reason == "tool_calls"
    
    # Verify the Bedrock API was called with the correct tools format
    mock_client.converse.assert_called_once()
    call_args = mock_client.converse.call_args[1]
    assert "tools" in call_args
    assert len(call_args["tools"]) == 1
    assert call_args["tools"][0]["name"] == "get_weather"
    assert "inputSchema" in call_args["tools"][0]
    assert call_args["tools"][0]["inputSchema"]["json"]["properties"]["location"]["type"] == "string"


def test_aws_normalize_response():
    """Test the normalize_response method."""
    provider = AwsChatProvider()
    
    # Test response with no tool calls
    basic_response = {
        "output": {
            "message": {
                "content": [{"text": "Basic response"}]
            }
        }
    }
    norm_response = provider.normalize_response(basic_response)
    assert norm_response.choices[0].message.content == "Basic response"
    assert not hasattr(norm_response.choices[0].message, "tool_calls")
    
    # Test response with tool calls
    tool_response = {
        "output": {
            "message": {
                "content": [{"text": "Response with tool"}],
                "toolUse": [
                    {
                        "toolName": "search",
                        "input": json.dumps({"query": "test query"})
                    }
                ]
            }
        }
    }
    norm_response = provider.normalize_response(tool_response)
    assert norm_response.choices[0].message.content == "Response with tool"
    assert norm_response.choices[0].message.tool_calls is not None
    assert norm_response.choices[0].message.tool_calls[0]["function"]["name"] == "search"
    assert json.loads(norm_response.choices[0].message.tool_calls[0]["function"]["arguments"])["query"] == "test query"
    assert norm_response.choices[0].finish_reason == "tool_calls"