"""
Model client abstraction layer for the AI participant experiment.

Provides a uniform interface over different provider APIs so the experiment runner
does not need to know which provider it is talking to. Each client takes a list of
messages in OpenAI-style format and returns the assistant's text response.

To add a new provider, subclass ModelClient and implement chat().
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Optional


class ModelClient(ABC):
    """
    Abstract base class for all model clients.

    Arguments:
        • model_name:
            The provider-specific model identifier string.
        • temperature:
            Sampling temperature. Use 1.0 for maximum within-model variance.
        • max_tokens:
            Maximum tokens the model may generate per turn.
    """

    def __init__(self, model_name: str, temperature: float = 1.0, max_tokens: int = 1024):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def chat(self, messages: list[dict]) -> str:
        """
        Sends a conversation to the model and returns the assistant's text reply.

        Arguments:
            • messages:
                List of dicts with keys "role" ("system" | "user" | "assistant") and "content".

        Returns:
            • The assistant's text response as a plain string.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r}, temperature={self.temperature})"


class ClaudeClient(ModelClient):
    """
    Client for Anthropic Claude models via the anthropic Python SDK.

    Requires the ANTHROPIC_API_KEY environment variable to be set,
    or an api_key argument passed directly.

    Arguments:
        • model_name:
            Anthropic model ID, e.g. "claude-sonnet-4-6".
        • temperature:
            Sampling temperature (0.0–1.0).
        • max_tokens:
            Maximum tokens per response.
        • api_key:
            Optional API key. If None, reads from ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-6",
        temperature: float = 1.0,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name, temperature, max_tokens)
        self._api_key = api_key

    async def chat(self, messages: list[dict]) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)

        "Separate system message (if present) from the user/assistant turns"
        system_text = ""
        conversation_messages = []
        for message in messages:
            if message["role"] == "system":
                system_text = message["content"]
            else:
                conversation_messages.append(message)

        kwargs = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": conversation_messages,
        }
        if system_text:
            kwargs["system"] = system_text

        response = await client.messages.create(**kwargs)
        return response.content[0].text


class OpenAIClient(ModelClient):
    """
    Client for OpenAI models via the openai Python SDK.

    Requires the OPENAI_API_KEY environment variable to be set,
    or an api_key argument passed directly.

    Arguments:
        • model_name:
            OpenAI model ID, e.g. "gpt-4o", "gpt-4o-mini", "o3-mini".
        • temperature:
            Sampling temperature (0.0–2.0 for most models).
        • max_tokens:
            Maximum tokens per response.
        • api_key:
            Optional API key. If None, reads from OPENAI_API_KEY env var.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 1.0,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name, temperature, max_tokens)
        self._api_key = api_key

    async def chat(self, messages: list[dict]) -> str:
        import openai

        client = openai.AsyncOpenAI(api_key=self._api_key)

        response = await client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


class GeminiClient(ModelClient):
    """
    Client for Google Gemini models via the google-genai Python SDK.

    Install: pip install google-genai
    Requires the GOOGLE_API_KEY environment variable to be set,
    or an api_key argument passed directly.

    Arguments:
        • model_name:
            Gemini model ID, e.g. "gemini-2.5-flash", "gemini-2.0-flash".
        • temperature:
            Sampling temperature (0.0–2.0).
        • max_tokens:
            Maximum tokens per response.
        • api_key:
            Optional API key. If None, reads from GOOGLE_API_KEY env var.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 1.0,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name, temperature, max_tokens)
        self._api_key = api_key

    async def chat(self, messages: list[dict]) -> str:
        from google import genai
        from google.genai import types
        import os

        api_key = self._api_key or os.environ.get("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)

        "Convert OpenAI-style messages to Gemini format"
        system_text = ""
        gemini_contents = []

        for message in messages:
            if message["role"] == "system":
                system_text = message["content"]
            elif message["role"] == "user":
                gemini_contents.append(
                    types.Content(role="user", parts=[types.Part(text=message["content"])])
                )
            elif message["role"] == "assistant":
                gemini_contents.append(
                    types.Content(role="model", parts=[types.Part(text=message["content"])])
                )

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            system_instruction=system_text or None,
        )

        response = await client.aio.models.generate_content(
            model=self.model_name,
            contents=gemini_contents,
            config=config,
        )
        return response.text


class OllamaClient(ModelClient):
    """
    Client for locally-running models via Ollama (http://localhost:11434).

    No API key required. Requires Ollama to be running locally with the target model pulled.
    Run: ollama pull mistral (or whichever model you want)
    16GB RAM supports models up to ~8B parameters. Suggested: "mistral", "llama3.1:8b".

    Arguments:
        • model_name:
            Ollama model identifier, e.g. "mistral", "llama3.1:8b", "qwen2.5:7b".
        • temperature:
            Sampling temperature.
        • max_tokens:
            Maximum tokens per response.
        • base_url:
            Ollama server URL. Defaults to http://localhost:11434.
    """

    def __init__(
        self,
        model_name: str = "mistral",
        temperature: float = 1.0,
        max_tokens: int = 1024,
        base_url: str = "http://localhost:11434",
    ):
        super().__init__(model_name, temperature, max_tokens)
        self._base_url = base_url

    async def chat(self, messages: list[dict]) -> str:
        import json
        import aiohttp

        payload = {
            "model": self.model_name,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "stream": False,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}/api/chat",
                json=payload,
            ) as response:
                result = await response.json()
                if "error" in result:
                    raise RuntimeError(f"Ollama error: {result['error']}")
                if "message" not in result:
                    raise RuntimeError(f"Ollama returned unexpected response: {result}")
                return result["message"]["content"]


class GrokClient(ModelClient):
    """
    Client for xAI Grok models. Uses an OpenAI-compatible API.

    Requires the XAI_API_KEY environment variable, or an api_key argument.
    Get a key at console.x.ai (new accounts receive $25 in free credits).

    Arguments:
        • model_name:
            xAI model ID, e.g. "grok-3", "grok-2-1212", "grok-3-mini".
        • temperature:
            Sampling temperature.
        • max_tokens:
            Maximum tokens per response.
        • api_key:
            Optional API key. If None, reads from XAI_API_KEY env var.
    """

    def __init__(
        self,
        model_name: str = "grok-3",
        temperature: float = 1.0,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name, temperature, max_tokens)
        self._api_key = api_key

    async def chat(self, messages: list[dict]) -> str:
        import openai
        import os

        resolved_api_key = self._api_key or os.environ.get("XAI_API_KEY")
        client = openai.AsyncOpenAI(
            api_key=resolved_api_key,
            base_url="https://api.x.ai/v1",
        )
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


class DeepSeekClient(ModelClient):
    """
    Client for DeepSeek models. Uses an OpenAI-compatible API.

    Requires the DEEPSEEK_API_KEY environment variable, or an api_key argument.
    Get a key at platform.deepseek.com (very cheap: ~$0.14/1M input tokens for DeepSeek-V3).

    Arguments:
        • model_name:
            DeepSeek model ID, e.g. "deepseek-chat" (DeepSeek-V3), "deepseek-reasoner" (R1).
        • temperature:
            Sampling temperature.
        • max_tokens:
            Maximum tokens per response.
        • api_key:
            Optional API key. If None, reads from DEEPSEEK_API_KEY env var.
    """

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        temperature: float = 1.0,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name, temperature, max_tokens)
        self._api_key = api_key

    async def chat(self, messages: list[dict]) -> str:
        import openai
        import os

        resolved_api_key = self._api_key or os.environ.get("DEEPSEEK_API_KEY")
        client = openai.AsyncOpenAI(
            api_key=resolved_api_key,
            base_url="https://api.deepseek.com",
        )
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


PROVIDER_MAX_TEMPERATURE: dict[str, float] = {
    "claude": 1.0,
    "openai": 2.0,
    "gemini": 2.0,
    "grok": 2.0,
    "deepseek": 2.0,
    "ollama": 2.0,
}

MODEL_TO_PROVIDER: dict[str, str] = {
    "claude-sonnet-4-6": "claude",
    "claude-opus-4-7": "claude",
    "claude-haiku-4-5-20251001": "claude",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "o3-mini": "openai",
    "gemini-2.5-flash": "gemini",
    "gemini-2.5-pro": "gemini",
    "gemini-2.0-flash": "gemini",
    "gemini-1.5-pro": "gemini",
    "grok-3": "grok",
    "grok-3-mini": "grok",
    "deepseek-chat": "deepseek",
    "deepseek-reasoner": "deepseek",
    "mistral": "ollama",
    "llama3.1:8b": "ollama",
    "qwen2.5:7b": "ollama",
}


def get_client_for_model(model_name: str, **kwargs) -> ModelClient:
    """
    Auto-detects the provider from the model name and returns the appropriate ModelClient.

    Arguments:
        • model_name:
            A model identifier string from MODEL_TO_PROVIDER.
        • **kwargs:
            Additional arguments passed to the client constructor
            (e.g. temperature, max_tokens, api_key).

    Returns:
        • A ModelClient instance ready to accept chat() calls.
    """
    if model_name not in MODEL_TO_PROVIDER:
        raise ValueError(
            f"Unknown model {model_name!r}. Known models: {list(MODEL_TO_PROVIDER.keys())}"
        )

    provider_name = MODEL_TO_PROVIDER[model_name]
    return get_client(provider=provider_name, model_name=model_name, **kwargs)


def get_client(provider: str, model_name: Optional[str] = None, **kwargs) -> ModelClient:
    """
    Factory function that returns the appropriate ModelClient for a given provider.

    Arguments:
        • provider:
            One of "claude", "openai", "gemini", "ollama".
        • model_name:
            Provider-specific model identifier. If None, uses a sensible default.
        • **kwargs:
            Additional arguments passed to the client constructor
            (e.g. temperature, max_tokens, api_key).

    Returns:
        • A ModelClient instance ready to accept chat() calls.
    """
    provider_lower = provider.strip().lower()

    default_model_names = {
        "claude": "claude-sonnet-4-6",
        "openai": "gpt-4o",
        "gemini": "gemini-2.0-flash",
        "grok": "grok-3",
        "deepseek": "deepseek-chat",
        "ollama": "mistral",
    }

    if provider_lower not in default_model_names:
        raise ValueError(
            f"Unknown provider {provider!r}. Choose from: {list(default_model_names.keys())}"
        )

    resolved_model_name = model_name or default_model_names[provider_lower]

    provider_client_classes = {
        "claude": ClaudeClient,
        "openai": OpenAIClient,
        "gemini": GeminiClient,
        "grok": GrokClient,
        "deepseek": DeepSeekClient,
        "ollama": OllamaClient,
    }

    return provider_client_classes[provider_lower](model_name=resolved_model_name, **kwargs)
