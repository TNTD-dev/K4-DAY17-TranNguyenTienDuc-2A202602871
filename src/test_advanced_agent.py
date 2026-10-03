"""Step 6 integration checks using temporary profiles and fake live models."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import agent_advanced
from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import estimate_tokens
from model_provider import ProviderConfig


ROOT = Path(__file__).resolve().parent.parent


def config_for(tmp_path, threshold=1000):
    return LabConfig(
        base_dir=tmp_path, data_dir=ROOT / "data", state_dir=tmp_path / "state",
        compact_threshold_tokens=threshold, compact_keep_messages=4,
    )


def test_cross_thread_recall_compared_with_baseline(tmp_path):
    config = config_for(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    for agent in (advanced, baseline):
        agent.reply("user", "first", "Mình tên là An. Mình ở Huế. Mình đang làm data engineer.")
    question = "Nhắc lại tên, nơi ở và nghề nghiệp hiện tại của mình."
    answer = advanced.reply("user", "new", question)["answer"]
    forgotten = baseline.reply("user", "new", question)["answer"]
    assert all(value in answer for value in ("An", "Huế", "data engineer"))
    assert all(value not in forgotten for value in ("An", "Huế", "data engineer"))
    assert advanced.memory_file_size("user") > 0


def test_profile_survives_a_new_python_process(tmp_path):
    agent = AdvancedAgent(config_for(tmp_path), force_offline=True)
    agent.reply("user", "first", "Mình tên là An. Mình ở Huế.")
    script = (
        "import sys; from pathlib import Path; sys.path.insert(0, 'src'); "
        "from agent_advanced import AdvancedAgent; from config import LabConfig; "
        "agent=AdvancedAgent(LabConfig(state_dir=Path(sys.argv[1])),force_offline=True); "
        "print(agent.reply('user','new','Mình tên gì và đang ở đâu?')['answer'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(agent.config.state_dir)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert "An" in result.stdout and "Huế" in result.stdout


def test_current_profile_beats_old_thread_and_summary(tmp_path):
    agent = AdvancedAgent(config_for(tmp_path), force_offline=True)
    agent.reply("user", "old", "Mình ở Huế.")
    agent.reply("user", "other", "Mình ở Đà Nẵng.")
    agent.compact_memory.state["old"]["summary"] = "- location: Huế"
    answer = agent.reply("user", "old", "Hiện tại mình đang ở đâu?")["answer"]
    assert "Đà Nẵng" in answer and "Huế" not in answer
    assert agent.profile_store.read_text("user").count("- location:") == 1


def test_users_and_threads_are_isolated(tmp_path):
    agent = AdvancedAgent(config_for(tmp_path), force_offline=True)
    agent.reply("a", "a-thread", "Mình tên là An.")
    answer = agent.reply("b", "b-thread", "Mình tên gì?")["answer"]
    assert "An" not in answer
    assert agent.memory_file_size("b") == 0
    before = agent.token_usage("a-thread")
    with pytest.raises(ValueError, match="different user"):
        agent.reply("b", "a-thread", "Mình tên gì?")
    assert agent.token_usage("a-thread") == before


def test_questions_and_noise_do_not_rewrite_profile(tmp_path):
    agent = AdvancedAgent(config_for(tmp_path), force_offline=True)
    agent.reply("user", "a", "Mình ở Đà Nẵng. Mình đang làm MLOps engineer.")
    path = agent.profile_store.path_for("user")
    original = path.read_bytes()
    for message in (
        "Hiện tại mình đang ở đâu?", "Bạn biết DũngCT là ai không?",
        "Mình đùa rằng giờ chuyển sang product manager.",
        "Mình ở Hà Nội để họp hai ngày.",
    ):
        agent.reply("user", "new", message)
    assert path.read_bytes() == original


def test_historical_location_and_third_party_name_preserve_current_profile(tmp_path):
    agent = AdvancedAgent(config_for(tmp_path), force_offline=True)
    agent.reply("user", "setup", "Mình tên là Đức. Mình ở Huế.")
    agent.reply("user", "correction", "Mình ở Đà Nẵng, trước đây mình ở Huế.")
    agent.reply("user", "noise", "Bạn tôi tên là An.")
    answer = agent.reply("user", "recall", "Nhắc lại tên và nơi ở của mình.")["answer"]
    assert "Đức" in answer and "Đà Nẵng" in answer
    assert "Huế" not in answer and "An" not in answer
    facts = agent.profile_store.facts("user")
    assert facts["name"] == "Đức" and facts["location"] == "Đà Nẵng"


def test_partial_style_reaffirmation_and_explicit_changes(tmp_path):
    agent = AdvancedAgent(config_for(tmp_path), force_offline=True)
    agent.reply("user", "a", "Mình muốn bạn trả lời ngắn gọn thành 3 bullet và ưu tiên trade-off.")
    agent.reply("user", "a", "Mình muốn bạn trả lời ngắn gọn.")
    style = agent.profile_store.facts("user")["response_style"]
    assert "3 bullet" in style and "trade-off" in style
    response = agent.reply("user", "new", "Nhắc lại style trả lời mình thích.")["answer"]
    assert len(response.splitlines()) == 3
    assert all(line.startswith("- ") for line in response.splitlines())
    agent.reply("user", "a", "Mình muốn bạn trả lời chi tiết và không dùng bullet.")
    style = agent.profile_store.facts("user")["response_style"]
    assert "chi tiết" in style and "không dùng bullet" in style
    assert "3 bullet" not in style and "ngắn gọn" not in style
    assert not agent.reply("user", "last", "Nhắc lại style trả lời mình thích.")["answer"].startswith("- ")


def test_baseline_and_advanced_use_the_same_responder_with_identical_facts(tmp_path):
    config = config_for(tmp_path)
    agents = [BaselineAgent(config, force_offline=True), AdvancedAgent(config, force_offline=True)]
    for message in (
        "Mình tên là An. Mình muốn trả lời ngắn gọn thành 3 bullet.",
        "Mình muốn trả lời ngắn gọn.", "Mình tên gì và thích kiểu trả lời như thế nào?",
    ):
        replies = [agent.reply("user", "thread", message) for agent in agents]
        assert replies[0]["answer"] == replies[1]["answer"]
        assert replies[0]["agent_tokens"] == replies[1]["agent_tokens"]


def test_prompt_accounting_includes_profile_summary_and_recent_messages(tmp_path):
    config = config_for(tmp_path, 10000)
    agent = AdvancedAgent(config, force_offline=True)
    message = "Mình tên là An."
    reply = agent.reply("user", "thread", message)
    recent = agent.compact_memory.context("thread")["messages"]
    expected = estimate_tokens(agent._system_prompt("user", "thread")) + estimate_tokens(message)
    assert reply["prompt_tokens"] == expected
    assert recent[-1]["content"] == reply["answer"]
    before = agent._estimate_prompt_context_tokens("user", "thread")
    agent.compact_memory.state["thread"]["summary"] = "Nội dung lịch sử. " * 100
    assert agent._estimate_prompt_context_tokens("user", "thread") > before
    before = agent._estimate_prompt_context_tokens("user", "thread")
    agent.profile_store.write_text("user", agent.profile_store.read_text("user") + "Ghi chú. " * 100)
    assert agent._estimate_prompt_context_tokens("user", "thread") > before


def test_cumulative_counters_are_thread_local_and_unknown_threads_return_zero(tmp_path):
    agent = AdvancedAgent(config_for(tmp_path), force_offline=True)
    replies = [agent.reply("user", "a", text) for text in ("Mình tên là An.", "Mình tên gì?")]
    agent.reply("user", "b", "Hello")
    assert agent.token_usage("a") == sum(r["agent_tokens"] for r in replies)
    assert agent.prompt_token_usage("a") == sum(r["prompt_tokens"] for r in replies)
    assert agent.token_usage("unused") == agent.prompt_token_usage("unused") == 0
    assert agent.compaction_count("unused") == 0


def test_compact_reduces_prompt_load_without_losing_persistent_facts(tmp_path):
    config = config_for(tmp_path, 300)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    for agent in (advanced, baseline):
        agent.reply("user", "long", "Mình tên là An. Mình ở Huế.")
        for _ in range(30):
            agent.reply("user", "long", "Nội dung ngữ cảnh tạm thời rất dài. " * 20)
    assert advanced.compaction_count("long") > 1
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")
    assert "An" in advanced.reply("user", "new", "Mình tên gì?")["answer"]


def test_offline_runs_are_deterministic_on_clean_state(tmp_path):
    agents = [AdvancedAgent(config_for(tmp_path / suffix), force_offline=True) for suffix in ("a", "b")]
    for thread, text in [("a", "Mình tên là An."), ("b", "Mình tên gì?")]:
        assert agents[0].reply("user", thread, text) == agents[1].reply("user", thread, text)


def test_force_offline_and_missing_credentials_skip_model_construction(tmp_path, monkeypatch):
    def deny_build(*args, **kwargs):
        pytest.fail("Offline selection must happen before model construction.")

    monkeypatch.setattr(agent_advanced, "build_chat_model", deny_build)
    no_key = AdvancedAgent(config_for(tmp_path / "no-key"))
    with_key = config_for(tmp_path / "key")
    with_key.model = ProviderConfig("openai", "fake", 0.0, "placeholder")
    forced = AdvancedAgent(with_key, force_offline=True)
    assert no_key.langchain_agent is forced.langchain_agent is None
    assert forced.reply("user", "a", "Hello")["answer"]


def live_agent(tmp_path, monkeypatch, responses, callbacks=None, threshold=1000):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class FakeWithTools(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = FakeWithTools(responses=responses, callbacks=callbacks or [])
    monkeypatch.setattr(agent_advanced, "build_chat_model", lambda config: model)
    config = config_for(tmp_path, threshold)
    config.model = ProviderConfig("openai", "fake", 0.0, "placeholder")
    return AdvancedAgent(config)


def test_live_prompt_loads_profile_in_new_thread_without_other_users_history(tmp_path, monkeypatch):
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import AIMessage

    seen = []

    class Capture(BaseCallbackHandler):
        def on_chat_model_start(self, serialized, messages, **kwargs):
            seen.append([m.content for m in messages[0]])

    responses = [AIMessage(content="Reply", usage_metadata={
        "input_tokens": 90, "output_tokens": 8, "total_tokens": 98,
    }) for _ in range(3)]
    agent = live_agent(tmp_path, monkeypatch, responses, [Capture()])
    agent.reply("a", "first", "Mình tên là An.")
    result = agent.reply("a", "new", "Mình tên gì?")
    agent.reply("b", "other-user", "Hello")
    assert "- name: An" in seen[1][0]
    assert "Mình tên là An." not in seen[1][1:]
    assert all("An" not in text for text in seen[2])
    assert result["agent_tokens"] == 8 and result["prompt_tokens"] == 90


def test_live_graph_receives_compacted_history_instead_of_full_transcript(tmp_path, monkeypatch):
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import AIMessage

    seen = []

    class Capture(BaseCallbackHandler):
        def on_chat_model_start(self, serialized, messages, **kwargs):
            seen.append(messages[0])

    agent = live_agent(tmp_path, monkeypatch, [AIMessage(content="Reply") for _ in range(10)], [Capture()], 100)
    for i in range(10):
        agent.reply("user", "long", f"Message {i}: " + "context " * 100)
    assert agent.compaction_count("long") > 1
    assert len(seen[-1]) == agent.config.compact_keep_messages + 1  # plus system prompt
    assert len(agent.langchain_agent.get_state({"configurable": {"thread_id": "long"}}).values["messages"]) <= 5


@pytest.mark.parametrize("tool_name,args,expected", [
    ("read_user_profile", {}, "- name: An"),
    ("update_user_profile", {"key": "name", "value": "An"}, "Đã cập nhật"),
    ("update_user_profile", {"key": "location", "value": "Hà Nội"}, "Không ghi"),
])
def test_live_profile_tools_are_scoped_and_all_model_calls_are_counted(tmp_path, monkeypatch, tool_name, args, expected):
    from langchain_core.messages import AIMessage

    responses = [
        AIMessage(content="", tool_calls=[{"id": "call-1", "name": tool_name, "args": args}],
                  usage_metadata={"input_tokens": 80, "output_tokens": 7, "total_tokens": 87}),
        AIMessage(content="Final reply", usage_metadata={"input_tokens": 100, "output_tokens": 5, "total_tokens": 105}),
    ]
    agent = live_agent(tmp_path, monkeypatch, responses)
    reply = agent.reply("user", "thread", "Mình tên là An.")
    state = agent.langchain_agent.get_state({"configurable": {"thread_id": "thread"}}).values
    tool_messages = [m for m in state["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1 and expected in tool_messages[0].content
    assert agent.profile_store.facts("user") == {"name": "An"}
    assert reply["agent_tokens"] == 12 and reply["prompt_tokens"] == 180


def test_live_fallback_counts_the_context_when_metadata_is_missing(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage

    agent = live_agent(tmp_path, monkeypatch, [AIMessage(content="Reply without metadata")])
    reply = agent.reply("user", "thread", "Mình tên là An.")
    assert reply["agent_tokens"] == estimate_tokens(reply["answer"])
    assert reply["prompt_tokens"] > estimate_tokens("Mình tên là An.")


def test_live_fallback_counts_generated_tool_arguments(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage

    responses = [
        AIMessage(content="", tool_calls=[{"id": "call-1", "name": "read_user_profile", "args": {}}]),
        AIMessage(content="Final reply"),
    ]
    agent = live_agent(tmp_path, monkeypatch, responses)
    reply = agent.reply("user", "thread", "Mình tên là An.")
    assert reply["agent_tokens"] > estimate_tokens("Final reply")
    assert reply["prompt_tokens"] > 2 * estimate_tokens(agent._system_prompt("user", "thread"))


def test_supplied_datasets_recall_after_each_conversation_and_stress_saves_context(tmp_path):
    config = config_for(tmp_path)
    advanced, baseline = AdvancedAgent(config, force_offline=True), BaselineAgent(config, force_offline=True)
    for filename in ("conversations.json", "advanced_long_context.json"):
        for c in json.loads((config.data_dir / filename).read_text(encoding="utf-8")):
            for message in c["turns"]:
                for agent in (advanced, baseline):
                    agent.reply(c["user_id"], c["id"], message)
            for i, question in enumerate(c["recall_questions"]):
                reply = advanced.reply(c["user_id"], f"recall-{c['id']}-{i}", question["question"])
                assert all(value.casefold() in reply["answer"].casefold() for value in question["expected_contains"])
    assert advanced.compaction_count("stress-01") > 1
    assert advanced.prompt_token_usage("stress-01") < baseline.prompt_token_usage("stress-01")
    facts = advanced.profile_store.facts("dungct_stress")
    assert facts["location"] == "Đà Nẵng" and facts["profession"] == "MLOps engineer"
    assert "3 bullet" in facts["response_style"] and "trade-off" in facts["response_style"]
