from unittest.mock import MagicMock, patch

import pytest
import json
from langchain.chat_models import ChatOpenAI
from langchain.schema import LLMResult, Generation, ChatGeneration, AIMessage
from langchain.schema.messages import HumanMessage, SystemMessage

from aisuite.providers.langchain_provider import LangchainChatProvider
from aisuite.provider import LLMError


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")


def test_langchain_provider_init():
    """Test that the provider is initialized correctly."""
    # Test with API key from environment
    provider = LangchainChatProvider()
    assert provider.api_key == "test-api-key"
    
    # Test with explicit API key
    provider = LangchainChatProvider(api_key="explicit-api-key")
    assert provider.api_key == "explicit-api-key"
    
    # Test with other configuration options
    provider = LangchainChatProvider(
        api_key="test-key",
        base_url="https://custom-openai.example.com",
        timeout=120,
        max_retries=5,
        model="gpt-4"
    )
    assert provider.api_key == "test-key"
    assert provider.base_url == "https://custom-openai.example.com"
    assert provider.timeout == 120
    assert provider.max_retries == 5
    assert provider.default_model == "gpt-4"


def test_chat_completions_create_basic():
    """Test basic chat completions without tools."""
    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "gpt-3.5-turbo"
    chosen_temperature = 0.75
    response_text_content = "Hello! How can I help you today?"

    # Create mock AIMessage with content
    ai_message = AIMessage(content=response_text_content)
    
    # Create mock generation
    mock_generation = ChatGeneration(message=ai_message)
    mock_generation.finish_reason = "stop"
    
    # Create mock LLMResult
    mock_llm_result = LLMResult(
        generations=[[mock_generation]],
        llm_output={"id": "test-id", "created": 1234567890, "model": selected_model}
    )

    provider = LangchainChatProvider()
    
    # Mock the ChatOpenAI.generate method
    with patch.object(
        ChatOpenAI,
        'generate',
        return_value=mock_llm_result
    ) as mock_generate:
        response = provider.chat_completions_create(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        # Check that generate was called with the right messages
        mock_generate.assert_called_once()
        
        # Check response content
        assert response.choices[0].message.content == response_text_content
        assert response.model == selected_model
        assert response.choices[0].finish_reason == "stop"


def test_chat_completions_with_function_call():
    """Test chat completions with function calls."""
    message_history = [
        {"role": "user", "content": "What's the weather in New York?"}
    ]
    selected_model = "gpt-4"
    
    # Create mock AI message with function call
    ai_message = AIMessage(content="I'll check the weather for you.")
    ai_message.additional_kwargs = {
        "function_call": {
            "name": "get_weather",
            "arguments": json.dumps({"location": "New York"})
        }
    }
    
    # Create mock generation
    mock_generation = ChatGeneration(message=ai_message)
    mock_generation.finish_reason = "function_call"
    
    # Create mock LLMResult
    mock_llm_result = LLMResult(
        generations=[[mock_generation]],
        llm_output={"id": "test-id", "created": 1234567890, "model": selected_model}
    )
    
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

    provider = LangchainChatProvider()
    
    # Mock the ChatOpenAI.generate method
    with patch.object(
        ChatOpenAI,
        'generate',
        return_value=mock_llm_result
    ) as mock_generate:
        response = provider.chat_completions_create(
            messages=message_history,
            model=selected_model,
            tools=tools
        )

        # Check that the function call is in the response
        assert response.choices[0].message.function_call is not None
        assert response.choices[0].message.function_call["name"] == "get_weather"
        assert json.loads(response.choices[0].message.function_call["arguments"])["location"] == "New York"
        assert response.choices[0].finish_reason == "function_call"


def test_chat_completions_with_tool_calls():
    """Test chat completions with tool calls (new format)."""
    message_history = [
        {"role": "user", "content": "Book a flight to New York and reserve a hotel"}
    ]
    selected_model = "gpt-4"
    
    # Create mock AI message with tool calls
    ai_message = AIMessage(content="I'll help you book a flight and hotel.")
    ai_message.additional_kwargs = {
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "book_flight",
                    "arguments": json.dumps({"destination": "New York", "date": "2023-12-15"})
                }
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "book_hotel",
                    "arguments": json.dumps({"city": "New York", "check_in": "2023-12-15", "nights": 3})
                }
            }
        ]
    }
    
    # Create mock generation
    mock_generation = ChatGeneration(message=ai_message)
    mock_generation.finish_reason = "tool_calls"
    
    # Create mock LLMResult
    mock_llm_result = LLMResult(
        generations=[[mock_generation]],
        llm_output={"id": "test-id", "created": 1234567890, "model": selected_model}
    )
    
    # Define tools
    tools = [
        {
            "name": "book_flight", 
            "description": "Book a flight to a destination",
            "args": {
                "destination": {"type": "string", "description": "City name"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
            }
        },
        {
            "name": "book_hotel", 
            "description": "Book a hotel in a city",
            "args": {
                "city": {"type": "string", "description": "City name"},
                "check_in": {"type": "string", "description": "Check-in date"},
                "nights": {"type": "integer", "description": "Number of nights"}
            }
        }
    ]

    provider = LangchainChatProvider()
    
    # Mock the ChatOpenAI.generate method
    with patch.object(
        ChatOpenAI,
        'generate',
        return_value=mock_llm_result
    ) as mock_generate:
        response = provider.chat_completions_create(
            messages=message_history,
            model=selected_model,
            tools=tools
        )

        # Check that the tool calls are in the response
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 2
        assert response.choices[0].message.tool_calls[0]["function"]["name"] == "book_flight"
        assert response.choices[0].message.tool_calls[1]["function"]["name"] == "book_hotel"
        assert response.choices[0].finish_reason == "tool_calls"


def test_error_handling():
    """Test error handling in the provider."""
    provider = LangchainChatProvider()
    
    # Mock ChatOpenAI.generate to raise an exception
    with patch.object(
        ChatOpenAI,
        'generate',
        side_effect=Exception("Model connection failed")
    ):
        with pytest.raises(LLMError) as excinfo:
            provider.chat_completions_create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo"
            )
        
        assert "Error in Langchain chat completion" in str(excinfo.value)


def test_no_model_provided():
    """Test error when no model is provided."""
    provider = LangchainChatProvider()
    
    with pytest.raises(LLMError) as excinfo:
        provider.chat_completions_create(
            messages=[{"role": "user", "content": "Hello"}],
            model=None
        )
    
    assert "No model specified" in str(excinfo.value)


def test_message_conversion():
    """Test that messages are correctly converted to Langchain format."""
    provider = LangchainChatProvider()
    
    # Create messages of different types
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "How can I help?"},
        {"role": "function", "name": "get_time", "content": '{"time": "12:00"}'}
    ]
    
    # Mock ChatOpenAI to capture the messages
    mock_generate = MagicMock(return_value=LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="Hello"))]],
        llm_output={}
    ))
    
    with patch.object(ChatOpenAI, 'generate', mock_generate):
        provider.chat_completions_create(
            messages=messages,
            model="gpt-3.5-turbo"
        )
        
        # Get the messages passed to generate
        call_args = mock_generate.call_args[0][0]
        
        # Verify the message types
        assert isinstance(call_args[0], SystemMessage)
        assert isinstance(call_args[1], HumanMessage)
        assert isinstance(call_args[2], AIMessage)
        assert call_args[3].type == "function"
        assert call_args[3].name == "get_time"