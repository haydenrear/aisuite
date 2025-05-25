import google
import json

from aisuite.framework import ChatCompletionResponse
from aisuite.framework.tool_utils import SerializedTools


def transform_roles(messages):
    """Transform the roles in the messages based on the provided transformations."""
    openai_roles_to_google_roles = {
        "system": "user",
        "assistant": "model",
    }

    for message in messages:
        if role := openai_roles_to_google_roles.get(message["role"], None):
            message["role"] = role
    return messages


def normalize_response(response):
    """Normalize the response from Google AI to match OpenAI's response format."""
    openai_response = ChatCompletionResponse()
    
    # Extract the response content
    if hasattr(response.candidates[0].content.parts[0], 'text'):
        openai_response.choices[0].message.content = response.candidates[0].content.parts[0].text
    else:
        openai_response.choices[0].message.content = ""
    
    # Check if the response contains function calls
    if hasattr(response.candidates[0], 'function_calls') and response.candidates[0].function_calls:
        tool_calls = []
        for i, fc in enumerate(response.candidates[0].function_calls):
            tool_calls.append({
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": fc.name,
                    "arguments": fc.args
                }
            })
        
        if tool_calls:
            openai_response.choices[0].message.tool_calls = tool_calls
            openai_response.choices[0].finish_reason = "tool_calls"
    
    # Add additional response metadata if available
    if hasattr(response, 'usage'):
        openai_response.usage = response.usage
    
    return openai_response


def convert_openai_to_google_ai(messages):
    """Convert OpenAI messages to Google AI messages using vertex maybe if it talks like a duck. Looks the same"""
    if len(messages) == 0:
        return []

    history = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        # Create message with basic text content
        google_message = {'role': role, 'parts': [{'text': content}]}
        
        # Handle function calls in assistant messages
        if role == "assistant" and "function_call" in message:
            google_message['function_call'] = {
                'name': message["function_call"]["name"],
                'args': message["function_call"]["arguments"]
            }
        
        # Handle function results in function messages
        if role == "function":
            google_message['parts'][0]['function_response'] = {
                'name': message.get("name", ""),
                'response': content
            }
        
        history.append(google_message)
    return history
