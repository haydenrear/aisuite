import google

from aisuite.framework import ChatCompletionResponse


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
    openai_response.choices[0].message.content = (
        response.candidates[0].content.parts[0].text
    )
    return openai_response


def convert_openai_to_google_ai(messages):
    """Convert OpenAI messages to Google AI messages using vertex maybe if it talks like a duck. Looks the same"""
    if len(messages) == 0:
        return []

    history = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        history.append({'role': role, 'parts': [{'text': content}]})
    return history
