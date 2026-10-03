"""Step 5: baseline session behavior and accounting, without live API calls."""

import json
from pathlib import Path

import pytest

import agent_baseline
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import UserProfileStore, estimate_tokens
from model_provider import ProviderConfig
from offline_response import BASE_SYSTEM_PROMPT


def make_baseline(tmp_path, **kwargs):
    config = LabConfig(
        base_dir=tmp_path,
        data_dir=Path(__file__).resolve().parent.parent / "data",
        state_dir=tmp_path / "state",
    )
    return BaselineAgent(config, **kwargs)


def test_remembers_within_thread_but_forgets_in_new_thread(tmp_path):
    agent = make_baseline(tmp_path, force_offline=True)
    agent.reply("user-a", "thread-a", "Mình tên là An. Mình ở Huế.")
    remembered = agent.reply("user-a", "thread-a", "Mình tên gì và đang ở đâu?")["answer"]
    forgotten = agent.reply("user-a", "thread-b", "Mình tên gì và đang ở đâu?")["answer"]
    assert "An" in remembered and "Huế" in remembered
    assert "An" not in forgotten and "Huế" not in forgotten
    assert "chưa có thông tin" in forgotten


def test_latest_correction_is_used_within_thread(tmp_path):
    agent = make_baseline(tmp_path, force_offline=True)
    agent.reply("user", "thread", "Mình ở Huế và đang làm backend engineer.")
    agent.reply("user", "thread", "Mình ở Đà Nẵng. Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")
    answer = agent.reply("user", "thread", "Nhắc lại nơi ở và nghề nghiệp hiện tại của mình.")["answer"]
    assert "Đà Nẵng" in answer and "MLOps engineer" in answer
    assert "Huế" not in answer and "backend engineer" not in answer


def test_profile_store_is_never_read_or_written(tmp_path, monkeypatch):
    store = UserProfileStore(tmp_path / "state" / "profiles")
    path = store.write_text("user", "# User\n- name: SecretName\n")
    original = path.read_bytes()

    def deny_store_access(*args, **kwargs):
        pytest.fail("Baseline must not access persistent profiles.")

    monkeypatch.setattr(UserProfileStore, "read_text", deny_store_access)
    monkeypatch.setattr(UserProfileStore, "write_text", deny_store_access)
    agent = make_baseline(tmp_path, force_offline=True)
    answer = agent.reply("user", "new-thread", "Mình tên gì?")["answer"]
    assert "SecretName" not in answer
    assert path.read_bytes() == original
    assert list((tmp_path / "state").rglob("User.md")) == [path]


def test_fresh_agent_forgets_even_when_thread_id_is_reused(tmp_path):
    first = make_baseline(tmp_path, force_offline=True)
    first.reply("user", "thread", "Mình tên là An.")
    fresh = make_baseline(tmp_path, force_offline=True)
    assert "An" not in fresh.reply("user", "thread", "Mình tên gì?")["answer"]


def test_question_text_cannot_be_used_as_a_profile_fact(tmp_path):
    agent = make_baseline(tmp_path, force_offline=True)
    answer = agent.reply("user", "thread", "Bạn biết DũngCT là ai không? Nhắc lại tên mình.")["answer"]
    assert "DũngCT" not in answer


def test_per_turn_and_cumulative_token_accounting(tmp_path):
    agent = make_baseline(tmp_path, force_offline=True)
    first_message = "Mình tên là An."
    first = agent.reply("user", "thread", first_message)
    second_message = "Mình tên gì?"
    second = agent.reply("user", "thread", second_message)
    system_tokens = estimate_tokens(BASE_SYSTEM_PROMPT)
    assert first["prompt_tokens"] == system_tokens + estimate_tokens(first_message)
    assert second["prompt_tokens"] == (
        system_tokens + estimate_tokens(first_message)
        + estimate_tokens(first["answer"]) + estimate_tokens(second_message)
    )
    assert agent.token_usage("thread") == sum(estimate_tokens(r["answer"]) for r in (first, second))
    assert agent.prompt_token_usage("thread") == first["prompt_tokens"] + second["prompt_tokens"]
    assert len(agent.sessions["thread"].messages) == 4
    assert agent.token_usage("unused") == agent.prompt_token_usage("unused") == 0
    assert "unused" not in agent.sessions


def test_long_thread_keeps_all_messages_without_compaction(tmp_path):
    agent = make_baseline(tmp_path, force_offline=True)
    previous_prompt = 0
    for _ in range(20):
        reply = agent.reply("user", "thread", "Nội dung dài. " * 100)
        assert reply["prompt_tokens"] > previous_prompt
        previous_prompt = reply["prompt_tokens"]
    assert len(agent.sessions["thread"].messages) == 40
    assert agent.compaction_count("thread") == 0
    assert not list(tmp_path.rglob("User.md"))


def test_offline_run_is_deterministic(tmp_path):
    first = make_baseline(tmp_path / "a", force_offline=True)
    second = make_baseline(tmp_path / "b", force_offline=True)
    for message in ["Mình tên là An.", "Mình thích Python.", "Nhắc lại tên và mối quan tâm chính của mình."]:
        assert first.reply("user", "thread", message) == second.reply("user", "thread", message)


def test_same_thread_id_cannot_be_reused_by_another_user(tmp_path):
    agent = make_baseline(tmp_path, force_offline=True)
    agent.reply("user-a", "thread", "Mình tên là An.")
    before = agent.token_usage("thread")
    with pytest.raises(ValueError, match="different user"):
        agent.reply("user-b", "thread", "Mình tên gì?")
    assert agent.token_usage("thread") == before


@pytest.mark.parametrize("user_id,thread_id", [("", "thread"), ("user", " ")])
def test_empty_session_identifiers_are_rejected(tmp_path, user_id, thread_id):
    with pytest.raises(ValueError, match="empty"):
        make_baseline(tmp_path, force_offline=True).reply(user_id, thread_id, "Hello")


def test_force_offline_never_builds_model_even_with_credentials(tmp_path, monkeypatch):
    def deny_model_build(*args, **kwargs):
        pytest.fail("force_offline must not construct a model.")

    monkeypatch.setattr(agent_baseline, "build_chat_model", deny_model_build)
    config = LabConfig(state_dir=tmp_path, model=ProviderConfig("openai", "test-model", 0.0, "placeholder"))
    agent = BaselineAgent(config, force_offline=True)
    assert agent.langchain_agent is None
    assert agent.reply("user", "thread", "Hello")["answer"]


def test_missing_live_credentials_falls_back_to_offline(tmp_path, monkeypatch):
    def deny_model_build(*args, **kwargs):
        pytest.fail("Missing credentials should select offline before model construction.")

    monkeypatch.setattr(agent_baseline, "build_chat_model", deny_model_build)
    agent = make_baseline(tmp_path)
    assert agent.langchain_agent is None
    assert agent.reply("user", "thread", "Hello")["answer"]


def test_live_checkpointer_is_thread_local_and_usage_metadata_is_counted(tmp_path, monkeypatch):
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    seen = []

    class CaptureMessages(BaseCallbackHandler):
        def on_chat_model_start(self, serialized, messages, **kwargs):
            seen.append([m.content for m in messages[0]])

    fake = FakeMessagesListChatModel(
        responses=[AIMessage(content="Live reply", usage_metadata={
            "input_tokens": 80, "output_tokens": 7, "total_tokens": 87,
        }) for _ in range(3)], callbacks=[CaptureMessages()],
    )
    monkeypatch.setattr(agent_baseline, "build_chat_model", lambda config: fake)
    config = LabConfig(state_dir=tmp_path, model=ProviderConfig("openai", "fake", 0.0, "placeholder"))
    agent = BaselineAgent(config)
    assert agent.reply("user", "a", "Secret in thread A")["agent_tokens"] == 7
    agent.reply("user", "a", "Follow-up")
    agent.reply("user", "b", "New thread")
    assert "Secret in thread A" in seen[1]
    assert "Secret in thread A" not in seen[2]
    assert "Follow-up" not in seen[2]
    assert agent.token_usage("a") == 14
    assert agent.prompt_token_usage("a") == 160
    assert agent.prompt_token_usage("b") == 80


def test_live_without_usage_metadata_uses_context_estimates(tmp_path, monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    monkeypatch.setattr(agent_baseline, "build_chat_model", lambda config: FakeMessagesListChatModel(
        responses=[AIMessage(content="Reply without metadata")],
    ))
    config = LabConfig(state_dir=tmp_path, model=ProviderConfig("ollama", "fake", 0.0))
    agent = BaselineAgent(config)
    reply = agent.reply("user", "thread", "Hello")
    assert reply["agent_tokens"] == estimate_tokens(reply["answer"])
    assert reply["prompt_tokens"] == estimate_tokens(BASE_SYSTEM_PROMPT) + estimate_tokens("Hello")


def test_real_datasets_cross_session_recall_has_no_history_leak(tmp_path):
    agent = make_baseline(tmp_path, force_offline=True)
    for filename in ("conversations.json", "advanced_long_context.json"):
        for conversation in json.loads((agent.config.data_dir / filename).read_text(encoding="utf-8")):
            for message in conversation["turns"]:
                agent.reply(conversation["user_id"], conversation["id"], message)
            for index, question in enumerate(conversation["recall_questions"]):
                answer = agent.reply(
                    conversation["user_id"], f"recall-{conversation['id']}-{index}", question["question"],
                )["answer"]
                assert not any(value.casefold() in answer.casefold() for value in question["expected_contains"])
    assert not list(tmp_path.rglob("User.md"))
