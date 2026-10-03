from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

from model_provider import ProviderConfig, normalize_provider


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "local-model",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-4-5",
    "ollama": "llama3.2",
    "openrouter": "openai/gpt-4o-mini",
}
KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "custom": "CUSTOM_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


@dataclass
class LabConfig:
    """Shared paths, compaction settings and optional live-model configuration."""

    base_dir: Path = REPO_ROOT
    data_dir: Path = REPO_ROOT / "data"
    state_dir: Path = REPO_ROOT / "state"
    compact_threshold_tokens: int = 1000
    compact_keep_messages: int = 4
    model: ProviderConfig = field(
        default_factory=lambda: ProviderConfig("openai", DEFAULT_MODELS["openai"], 0.0)
    )
    judge_model: ProviderConfig = field(
        default_factory=lambda: ProviderConfig("openai", DEFAULT_MODELS["openai"], 0.0)
    )


def _positive_int(env: dict, name: str, default: int) -> int:
    try:
        value = int(env.get(name, default))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer.") from None
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _model_config(env: dict, prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    provider = normalize_provider(env.get(f"{prefix}_PROVIDER") or (fallback.provider if fallback else "openai"))
    same_provider = fallback is not None and provider == fallback.provider
    model_name = env.get(f"{prefix}_MODEL") or (fallback.model_name if same_provider else DEFAULT_MODELS[provider])
    temperature_name = f"{prefix}_TEMPERATURE"
    try:
        temperature = float(env.get(temperature_name) or (fallback.temperature if same_provider else 0.0))
    except (TypeError, ValueError):
        raise ValueError(f"{temperature_name} must be a finite number between 0 and 2.") from None
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError(f"{temperature_name} must be a finite number between 0 and 2.")
    api_key = env.get(f"{prefix}_API_KEY") or (fallback.api_key if same_provider else env.get(KEY_ENV[provider]))
    if provider == "gemini" and not api_key:
        api_key = env.get("GOOGLE_API_KEY")
    base_url = env.get(f"{prefix}_BASE_URL") or (
        fallback.base_url if same_provider else env.get(f"{provider.upper()}_BASE_URL")
    )
    if provider == "ollama" and not base_url:
        base_url = "http://localhost:11434"
    return ProviderConfig(provider, model_name, temperature, api_key, base_url)


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Read root/.env with process-env precedence, without constructing live models."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    env = {**dotenv_values(root / ".env"), **os.environ}
    threshold = _positive_int(env, "COMPACT_THRESHOLD_TOKENS", 1000)
    keep = _positive_int(env, "COMPACT_KEEP_MESSAGES", 4)
    model = _model_config(env, "LLM")
    judge = _model_config(env, "JUDGE", model)
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(root, root / "data", state_dir, threshold, keep, model, judge)
