"""Step 7 checks for fair evaluation, accounting and repeatable offline runs."""

import copy
import json
import unicodedata
from pathlib import Path

import pytest

import benchmark
from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig


ROOT = Path(__file__).resolve().parent.parent


def sample():
    return [
        {"id": "one", "user_id": "user", "turns": ["Mình ở Huế."],
         "recall_questions": [{"question": "Mình đang ở đâu?", "expected_contains": ["Huế"]}]},
        {"id": "two", "user_id": "user", "turns": ["Mình hiện ở Đà Nẵng."],
         "recall_questions": [{"question": "Mình đang ở đâu?", "expected_contains": ["Đà Nẵng"]}]},
    ]


@pytest.mark.parametrize("answer,expected,recall,quality", [
    ("", ["Huế"], 0.0, 0.0),
    ("ở Huế", ["Huế", "An", "engineer"], 0.5, 1 / 3),
    ("AN ở HUẾ", ["An", "Huế"], 1.0, 1.0),
    (unicodedata.normalize("NFD", "Huế"), ["Huế"], 1.0, 1.0),
    ("anything", [], 0.0, 0.0),
])
def test_scoring(answer, expected, recall, quality):
    assert benchmark.recall_points(answer, expected) == recall
    assert benchmark.heuristic_quality(answer, expected) == pytest.approx(quality)


@pytest.mark.parametrize("invalid", [
    {}, [None], [dict(sample()[0], id="")],
    [dict(sample()[0], turns="wrong")],
    [dict(sample()[0], recall_questions=[{"question": "?", "expected_contains": []}])],
    [sample()[0], sample()[0]],
])
def test_invalid_dataset_fails_before_evaluation(tmp_path, invalid):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError):
        benchmark.load_conversations(path)


def test_fair_threads_order_and_accounting(tmp_path):
    config = LabConfig(state_dir=tmp_path / "state")
    conversations = sample()
    original = copy.deepcopy(conversations)
    call_sequences = []
    for agent_type in (BaselineAgent, AdvancedAgent):
        agent = agent_type(config, force_offline=True)
        calls, results = [], []
        reply = agent.reply

        def tracked_reply(user_id, thread_id, message):
            calls.append((user_id, thread_id, message))
            result = reply(user_id, thread_id, message)
            results.append(result)
            return result

        agent.reply = tracked_reply
        row = benchmark.run_agent_benchmark(agent_type.__name__, agent, conversations, config)
        assert row.agent_tokens_only == sum(result["agent_tokens"] for result in results)
        assert row.prompt_tokens_processed == sum(result["prompt_tokens"] for result in results)
        assert row.recall_score == (1.0 if agent_type is AdvancedAgent else 0.0)
        assert row.response_quality == row.recall_score
        call_sequences.append(calls)
        with pytest.raises(ValueError, match="fresh"):
            benchmark.run_agent_benchmark("again", agent, conversations, config)
    assert call_sequences[0] == call_sequences[1]
    assert [call[1] for call in call_sequences[0]] == [
        "chat:one", "recall:one", "chat:two", "recall:two",
    ]
    assert conversations == original


def test_memory_growth_counts_unique_users_and_existing_bytes(tmp_path):
    config = LabConfig(state_dir=tmp_path / "state")
    agent = AdvancedAgent(config, force_offline=True)
    agent.profile_store.write_text("user", "# User Profile\n- name: An\n")
    initial = agent.memory_file_size("user")
    row = benchmark.run_agent_benchmark("Advanced", agent, sample(), config)
    assert row.memory_growth_bytes == agent.memory_file_size("user") - initial
    assert row.memory_growth_bytes > 0


def test_stress_metrics_include_compactions_and_prompt_savings(tmp_path):
    config = LabConfig(state_dir=tmp_path / "state")
    data = benchmark.load_conversations(ROOT / "data" / "advanced_long_context.json")
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    before = benchmark.run_agent_benchmark("Baseline", baseline, data, config)
    after = benchmark.run_agent_benchmark("Advanced", advanced, data, config)
    assert before.recall_score == 0 and after.recall_score == 1
    assert after.prompt_tokens_processed < before.prompt_tokens_processed
    assert after.compactions > 0
    assert after.compactions == sum(advanced.compaction_count(thread) for thread in (
        "chat:stress-01", "recall:stress-01",
    ))


def test_main_is_offline_repeatable_and_preserves_demo(tmp_path, monkeypatch, capsys):
    config = LabConfig(data_dir=ROOT / "data", state_dir=tmp_path / "state")
    demo = config.state_dir / "profiles" / "lab_demo" / "User.md"
    demo.parent.mkdir(parents=True)
    demo.write_text("demo profile", encoding="utf-8")
    monkeypatch.setattr(benchmark, "load_config", lambda _: config)

    def forbid_live_model(*args, **kwargs):
        raise AssertionError("Benchmark must not construct or call live models")

    monkeypatch.setattr("agent_baseline.build_chat_model", forbid_live_model)
    monkeypatch.setattr("agent_advanced.build_chat_model", forbid_live_model)
    benchmark.main()
    first = capsys.readouterr().out
    benchmark.main()
    assert capsys.readouterr().out == first
    assert first.count("| Baseline") == 2 and first.count("| Advanced") == 2
    assert "Standard Benchmark" in first and "Long-Context Stress Benchmark" in first
    for header in ("Agent tokens only", "Prompt tokens processed", "Cross-session recall",
                   "Response quality", "Memory growth (bytes)", "Compactions"):
        assert first.count(header) == 2
    assert demo.read_text(encoding="utf-8") == "demo profile"
    for name, user in (("standard", "dungct"), ("long_context", "dungct_stress")):
        assert (config.state_dir / "benchmarks" / name / "profiles" / user / "User.md").stat().st_size > 0


def test_empty_dataset_has_zero_metrics(tmp_path):
    config = LabConfig(state_dir=tmp_path / "state")
    row = benchmark.run_agent_benchmark("Baseline", BaselineAgent(config, True), [], config)
    assert row == benchmark.BenchmarkRow("Baseline", 0, 0, 0, 0, 0, 0)
