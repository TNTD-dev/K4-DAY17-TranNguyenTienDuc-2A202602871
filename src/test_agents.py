from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Use temporary files, stub models and a small compaction threshold."""
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=80,
        compact_keep_messages=2,
        model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
        judge_model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """UTF-8 profile edits persist on disk and preserve unrelated facts."""
    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / "profiles")
    assert store.read_text("user") == ""
    content = "# User Profile\n- name: An\n- location: Huế\n"
    path = store.write_text("user", content)
    assert path.name == "User.md" and path.is_relative_to(tmp_path)
    assert path.read_text(encoding="utf-8") == store.read_text("user") == content
    assert store.edit_text("user", "Huế", "Đà Nẵng") is True
    expected = content.replace("Huế", "Đà Nẵng")
    # A new store instance reads the edited file, rather than cached state.
    reopened = UserProfileStore(config.state_dir / "profiles")
    assert reopened.read_text("user") == expected
    assert reopened.file_size("user") == len(expected.encode("utf-8"))
    assert reopened.edit_text("user", "missing fact", "replacement") is False
    assert reopened.read_text("user") == expected


def test_compact_trigger(tmp_path: Path) -> None:
    """Crossing the threshold compresses old messages and keeps the recent tail."""
    config = make_config(tmp_path)
    memory = CompactMemoryManager(config.compact_threshold_tokens, config.compact_keep_messages)
    messages = [
        {"role": "user", "content": "Mình tên là An."},
        {"role": "assistant", "content": "Đã ghi nhận."},
        {"role": "user", "content": "Bản tin kỹ thuật dài để kiểm tra ngữ cảnh. " * 12},
    ]
    for item in messages[:2]:
        memory.append("long", item["role"], item["content"])
    assert memory.compaction_count("long") == 0
    memory.append("long", messages[2]["role"], messages[2]["content"])
    context = memory.context("long")
    assert memory.compaction_count("long") > 0
    assert context["messages"] == messages[-config.compact_keep_messages:]
    assert context["summary"]
    before = sum(estimate_tokens(item["content"]) for item in messages)
    after = estimate_tokens(context["summary"]) + sum(
        estimate_tokens(item["content"]) for item in context["messages"]
    )
    assert after < before
    assert memory.compaction_count("other-thread") == 0
    assert memory.context("other-thread")["messages"] == []


def test_cross_session_recall(tmp_path: Path) -> None:
    """Both remember within a thread; only Advanced remembers in a new one."""
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    statement = "Mình tên là An. Mình ở Huế. Mình đang làm data engineer."
    question = "Nhắc lại tên, nơi ở và nghề nghiệp hiện tại của mình."
    facts = ("An", "Huế", "data engineer")
    for agent in (baseline, advanced):
        agent.reply("user", "first-session", statement)
        answer = agent.reply("user", "first-session", question)["answer"]
        assert all(fact in answer for fact in facts)
    remembered = advanced.reply("user", "new-session", question)["answer"]
    forgotten = baseline.reply("user", "new-session", question)["answer"]
    assert all(fact in remembered for fact in facts)
    assert all(fact not in forgotten for fact in facts)
    assert advanced.memory_file_size("user") > 0
    profile = (config.state_dir / "profiles" / "user" / "User.md").read_text(encoding="utf-8")
    assert all(fact in profile for fact in facts)


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Actual prompt totals fall on a long thread when compaction is enabled."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    # Control: identical Advanced behavior with compaction disabled by threshold.
    uncompressed = AdvancedAgent(
        replace(config, state_dir=tmp_path / "uncompressed-state",
                compact_threshold_tokens=1_000_000),
        force_offline=True,
    )
    messages = ["Mình tên là An. Mình ở Huế."] + [
        f"Bản tin {index}: " + "Thông tin kỹ thuật tham khảo cho hội thoại. " * 20
        for index in range(16)
    ] + ["Nhắc lại tên và nơi ở của mình."]
    for message in messages:
        for agent in (baseline, advanced, uncompressed):
            agent.reply("user", "long-session", message)
    assert baseline.compaction_count("long-session") == 0
    assert advanced.compaction_count("long-session") > 0
    assert uncompressed.compaction_count("long-session") == 0
    assert 0 < advanced.prompt_token_usage("long-session") < baseline.prompt_token_usage("long-session")
    assert advanced.prompt_token_usage("long-session") < uncompressed.prompt_token_usage("long-session")
    # Savings must not come from producing shorter answers or losing user facts.
    assert advanced.token_usage("long-session") == uncompressed.token_usage("long-session")
    assert advanced.profile_store.read_text("user") == uncompressed.profile_store.read_text("user")
    for agent in (advanced, uncompressed):
        recalled = agent.reply("user", "recall-session", messages[-1])["answer"]
        assert "An" in recalled and "Huế" in recalled
