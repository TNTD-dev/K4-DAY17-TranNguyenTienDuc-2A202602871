from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProviderConfig:
    """Model settings; credentials are excluded from diagnostic repr output."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize provider names and reject unsupported choices early."""
    provider = value.strip().lower()
    aliases = {
        "anthorpic": "anthropic",
        "google": "gemini",
        "google-genai": "gemini",
        "openai-compatible": "custom",
        "open-router": "openrouter",
    }
    provider = aliases.get(provider, provider)
    supported = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")
    if provider not in supported:
        raise ValueError(f"Unsupported provider {value!r}; choose from {', '.join(supported)}.")
    return provider


def build_chat_model(config: ProviderConfig):
    """Construct a live model on demand; this does not invoke the model."""
    provider = normalize_provider(config.provider)
    if not config.model_name.strip():
        raise ValueError("model_name must not be empty.")
    if provider == "custom" and not config.base_url:
        raise ValueError("custom requires CUSTOM_BASE_URL or LLM_BASE_URL.")
    if provider not in ("ollama", "custom") and not config.api_key:
        raise ValueError(f"{provider} requires an API key for live mode; offline mode needs none.")

    kwargs = {"model": config.model_name, "temperature": config.temperature}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        if provider == "custom":
            # The OpenAI SDK requires a key even for unauthenticated local servers.
            kwargs.setdefault("api_key", "not-needed")
        return ChatOpenAI(**kwargs)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if config.base_url:
            kwargs.pop("base_url")
            kwargs["client_options"] = {"api_endpoint": config.base_url}
        return ChatGoogleGenerativeAI(**kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**kwargs)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs.pop("api_key", None)
        if config.api_key:
            kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {config.api_key}"}}
        return ChatOllama(**kwargs)

    from langchain_openrouter import ChatOpenRouter

    return ChatOpenRouter(**kwargs)
