import enum
import typing
from abc import ABC, abstractmethod
from pathlib import Path
import importlib
import os
import functools

from aisuite.framework.chat_provider import ChatProviderInterface
from aisuite.framework.chat_provider import ChatProvider
from aisuite.framework.embedding_provider import EmbeddingProviderInterface, EmbeddingProvider
from aisuite.framework.provider_interface import Provider, ProviderInterface


class ProviderType(enum.Enum):
    EMBEDDING = 'Embedding'
    CHAT = 'Chat'
    VALIDATION = 'Validation'
    RERANK = 'Rerank'

class LLMError(Exception):
    """Custom exception for LLM errors."""

    def __init__(self, message):
        super().__init__(message)


class ProviderFactory:
    """Factory to dynamically load provider instances based on naming conventions."""

    PROVIDERS_DIR = Path(__file__).parent / "providers"

    @classmethod
    def create_provider(cls, provider_key, config, provider_type: ProviderType = ProviderType.CHAT) \
            -> typing.Union[Provider, ProviderInterface]:
        """Dynamically load and create an instance of a provider based on the naming convention."""
        # Convert provider_key to the expected module and class names
        provider_class_name = f"{provider_key.capitalize()}{provider_type.value}Provider"
        provider_module_name = f"{provider_key}_provider"

        module_path = f"aisuite.providers.{provider_module_name}"

        # Lazily load the module
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Could not import module {module_path}: {str(e)}. Please ensure the provider is supported by doing ProviderFactory.get_supported_providers()"
            )
        except Exception as o:
            raise Exception(f"Could not import module for some other reason {str(o)}")


        # Instantiate the provider class
        provider_class = getattr(module, provider_class_name)
        return provider_class(**config)

    @classmethod
    def create_chat_provider(cls, provider_key, config) \
            -> typing.Union[ChatProviderInterface, ChatProvider]:
        """Dynamically load and create an instance of a provider based on the naming convention."""
        # Convert provider_key to the expected module and class names
        return typing.cast(typing.Union[ChatProviderInterface, ChatProvider],
                           cls.create_provider(provider_key, config, ProviderType.CHAT))

    @classmethod
    def create_embedding_provider(cls, provider_key, config) \
            -> typing.Union[EmbeddingProviderInterface, EmbeddingProvider]:
        """Dynamically load and create an instance of a provider based on the naming convention."""
        # Convert provider_key to the expected module and class names
        return typing.cast(typing.Union[EmbeddingProviderInterface, EmbeddingProvider],
                           cls.create_provider(provider_key, config, ProviderType.EMBEDDING))

    @classmethod
    def create_rerank_provider(cls, provider_key, config) \
            -> typing.Union[EmbeddingProviderInterface, EmbeddingProvider]:
        """Dynamically load and create an instance of a provider based on the naming convention."""
        # Convert provider_key to the expected module and class names
        return typing.cast(typing.Union[EmbeddingProviderInterface, EmbeddingProvider],
                           cls.create_provider(provider_key, config, ProviderType.RERANK))

    @classmethod
    @functools.cache
    def get_supported_providers(cls):
        """List all supported provider names based on files present in the providers directory."""
        provider_files = Path(cls.PROVIDERS_DIR).glob("*_provider.py")
        pr = [p for p in provider_files]
        return {file.stem.replace("_provider", "") for file in pr}

