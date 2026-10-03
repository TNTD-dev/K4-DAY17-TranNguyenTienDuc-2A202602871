"""Configuration/provider checks; never call a live model or require credentials."""

import os
import socket
from unittest.mock import patch

import pytest

from config import LabConfig, load_config
from model_provider import ProviderConfig, build_chat_model, normalize_provider


def test_offline_defaults_create_state_under_requested_root(tmp_path):
    with patch.dict(os.environ, {}, clear=True):
        config = load_config(tmp_path)
    assert config.base_dir == tmp_path.resolve()
    assert config.data_dir == tmp_path / "data"
    assert config.state_dir == tmp_path / "state"
    assert config.state_dir.is_dir()
    assert config.compact_threshold_tokens == 1000
    assert config.compact_keep_messages == 4
    assert config.model.api_key is None
    assert config.judge_model == config.model
    assert config.judge_model is not config.model


def test_dotenv_process_precedence_and_same_provider_judge(tmp_path):
    (tmp_path / ".env").write_text(
        "LLM_PROVIDER=anthorpic\nLLM_MODEL=from-file\n"
        "ANTHROPIC_API_KEY=test-placeholder\nCOMPACT_THRESHOLD_TOKENS=700\n"
        "COMPACT_KEEP_MESSAGES=2\nJUDGE_MODEL=judge-model\n",
        encoding="utf-8",
    )
    with patch.dict(os.environ, {"LLM_MODEL": "from-process"}, clear=True):
        config = load_config(tmp_path)
        assert "ANTHROPIC_API_KEY" not in os.environ
    assert config.model.provider == "anthropic"
    assert config.model.model_name == "from-process"
    assert config.model.api_key == "test-placeholder"
    assert config.judge_model.model_name == "judge-model"
    assert config.judge_model.api_key == config.model.api_key
    assert config.compact_threshold_tokens == 700
    assert config.compact_keep_messages == 2
    assert "test-placeholder" not in repr(config)


def test_judge_with_different_provider_uses_its_own_credentials(tmp_path):
    env = {
        "LLM_PROVIDER": "custom",
        "LLM_MODEL": "main-local",
        "CUSTOM_BASE_URL": "http://localhost:9000/v1",
        "CUSTOM_API_KEY": "main-placeholder",
        "JUDGE_PROVIDER": "gemini",
        "GEMINI_API_KEY": "judge-placeholder",
        "JUDGE_TEMPERATURE": "0.2",
    }
    with patch.dict(os.environ, env, clear=True):
        config = load_config(tmp_path)
    assert config.model.base_url == env["CUSTOM_BASE_URL"]
    assert config.judge_model.provider == "gemini"
    assert config.judge_model.model_name != config.model.model_name
    assert config.judge_model.api_key == "judge-placeholder"
    assert config.judge_model.base_url is None
    assert config.judge_model.temperature == 0.2


def test_generic_overrides_and_ollama_defaults(tmp_path):
    env = {
        "LLM_PROVIDER": "ollama",
        "LLM_API_KEY": "main-placeholder",
        "JUDGE_API_KEY": "judge-placeholder",
        "JUDGE_BASE_URL": "http://localhost:9001",
    }
    with patch.dict(os.environ, env, clear=True):
        config = load_config(tmp_path)
    assert config.model.base_url == "http://localhost:11434"
    assert config.model.api_key == "main-placeholder"
    assert config.judge_model.base_url == env["JUDGE_BASE_URL"]
    assert config.judge_model.api_key == "judge-placeholder"


def test_dataclass_model_defaults_are_independent():
    first, second = LabConfig(), LabConfig()
    first.model.model_name = "changed"
    assert second.model.model_name != "changed"
    assert first.judge_model.model_name != "changed"


@pytest.mark.parametrize("name,value", [
    ("COMPACT_THRESHOLD_TOKENS", "0"),
    ("COMPACT_THRESHOLD_TOKENS", "oops"),
    ("COMPACT_KEEP_MESSAGES", "-1"),
    ("LLM_TEMPERATURE", "nan"),
    ("JUDGE_TEMPERATURE", "inf"),
    ("LLM_TEMPERATURE", "3"),
])
def test_invalid_settings_fail_before_creating_state(tmp_path, name, value):
    with patch.dict(os.environ, {name: value}, clear=True):
        with pytest.raises(ValueError, match=name):
            load_config(tmp_path)
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("value,expected", [
    (" OPENAI ", "openai"),
    ("anthorpic", "anthropic"),
    ("Google", "gemini"),
    ("google-genai", "gemini"),
    ("openai-compatible", "custom"),
    ("open-router", "openrouter"),
])
def test_provider_aliases(value, expected):
    assert normalize_provider(value) == expected


def test_unsupported_provider_fails_early(tmp_path):
    with patch.dict(os.environ, {"LLM_PROVIDER": "unknown"}, clear=True):
        with pytest.raises(ValueError, match="Unsupported provider"):
            load_config(tmp_path)
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("provider,class_name", [
    ("openai", "ChatOpenAI"),
    ("custom", "ChatOpenAI"),
    ("gemini", "ChatGoogleGenerativeAI"),
    ("anthropic", "ChatAnthropic"),
    ("ollama", "ChatOllama"),
    ("openrouter", "ChatOpenRouter"),
])
def test_real_sdk_construction_without_network(monkeypatch, provider, class_name):
    def deny_network(*args, **kwargs):
        pytest.fail("Constructing a model must not make network requests.")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    base_url = "http://localhost:9000/v1"
    config = ProviderConfig(provider, "test-model", 0.1, "test-placeholder", base_url)
    with patch.dict(os.environ, {}, clear=True):
        model = build_chat_model(config)
    assert type(model).__name__ == class_name
    assert getattr(model, "model_name", getattr(model, "model", None)) == "test-model"
    assert model.temperature == 0.1
    if provider == "openai" or provider == "custom":
        assert model.openai_api_base == base_url
    elif provider == "anthropic":
        assert model.anthropic_api_url == base_url
    elif provider == "openrouter":
        assert model.openrouter_api_base == base_url
    else:
        actual_url = model.base_url
        assert actual_url == base_url or actual_url == {"api_endpoint": base_url}


@pytest.mark.parametrize("provider", ["openai", "gemini", "anthropic", "openrouter"])
def test_cloud_model_requires_key_only_when_built(provider):
    with pytest.raises(ValueError, match="API key"):
        build_chat_model(ProviderConfig(provider, "test-model", 0.0))


def test_custom_requires_url_but_can_use_unauthenticated_server():
    with pytest.raises(ValueError, match="BASE_URL"):
        build_chat_model(ProviderConfig("custom", "test-model", 0.0))
    model = build_chat_model(ProviderConfig("custom", "test-model", 0.0, base_url="http://localhost:9000/v1"))
    assert model.openai_api_key.get_secret_value() == "not-needed"
