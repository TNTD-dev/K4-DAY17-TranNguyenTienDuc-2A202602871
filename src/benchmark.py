from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from tabulate import tabulate

from config import LabConfig, load_config
from memory_store import UserProfileStore


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read and validate the dataset before either agent starts running."""
    conversations = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(conversations, list):
        raise ValueError(f"{path}: expected a list of conversations")
    ids = set()
    for index, conversation in enumerate(conversations):
        label = f"{path}: conversation {index}"
        if not isinstance(conversation, dict):
            raise ValueError(f"{label}: expected an object")
        for key in ("id", "user_id"):
            value = conversation.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label}: {key} must be a nonempty string")
        if conversation["id"] in ids:
            raise ValueError(f"{label}: duplicate conversation id")
        ids.add(conversation["id"])
        turns = conversation.get("turns")
        if not isinstance(turns, list) or any(
            not isinstance(turn, str) or not turn.strip() for turn in turns
        ):
            raise ValueError(f"{label}: turns must be a list of nonempty strings")
        questions = conversation.get("recall_questions")
        if not isinstance(questions, list):
            raise ValueError(f"{label}: recall_questions must be a list")
        for question in questions:
            if not isinstance(question, dict):
                raise ValueError(f"{label}: recall question must be an object")
            text = question.get("question")
            expected = question.get("expected_contains")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{label}: question must be a nonempty string")
            if not isinstance(expected, list) or not expected or any(
                not isinstance(item, str) or not item.strip() for item in expected
            ):
                raise ValueError(f"{label}: expected_contains must contain nonempty strings")
    return conversations


def _coverage(answer: str, expected: list[str]) -> float:
    if not expected:
        return 0.0
    normalized = unicodedata.normalize("NFC", answer).casefold()
    hits = sum(
        unicodedata.normalize("NFC", item.strip()).casefold() in normalized
        for item in expected if item.strip()
    )
    return hits / len(expected)


def recall_points(answer: str, expected: list[str]) -> float:
    """Zero for no matches, half for some matches, one for all matches."""
    coverage = _coverage(answer, expected)
    return 1.0 if coverage == 1.0 else 0.5 if coverage > 0 else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Fraction of expected facts found, using the same rule for both agents.

    This offline proxy does not judge fluency, negation or general reasoning.
    """
    return _coverage(answer, expected)


def run_agent_benchmark(
    agent_name: str, agent, conversations: list[dict[str, Any]], config: LabConfig,
) -> BenchmarkRow:
    """Evaluate a fresh agent, including chat AND recall turns in token totals.

    Recall follows each conversation, before later conversations can change facts.
    Callers supply a fresh instance; main also resets only suite-owned profiles.
    """
    if agent.config != config:
        raise ValueError("Agent and benchmark must use the same configuration")
    users = {conversation["user_id"] for conversation in conversations}
    memory_size = getattr(agent, "memory_file_size", lambda user_id: 0)
    initial_size = sum(memory_size(user_id) for user_id in users)
    agent_tokens = prompt_tokens = compactions = 0
    recall_scores = []
    quality_scores = []

    for conversation in conversations:
        user_id = conversation["user_id"]
        chat_thread = f"chat:{conversation['id']}"
        recall_thread = f"recall:{conversation['id']}"
        for thread in (chat_thread, recall_thread):
            if agent.prompt_token_usage(thread) or agent.token_usage(thread):
                raise ValueError("Benchmark requires fresh chat and recall threads")
        for message in conversation["turns"]:
            result = agent.reply(user_id, chat_thread, message)
            agent_tokens += result["agent_tokens"]
            prompt_tokens += result["prompt_tokens"]
        for question in conversation["recall_questions"]:
            # Expected answers are used only by the scorer, never by the agent.
            result = agent.reply(user_id, recall_thread, question["question"])
            agent_tokens += result["agent_tokens"]
            prompt_tokens += result["prompt_tokens"]
            recall_scores.append(recall_points(result["answer"], question["expected_contains"]))
            quality_scores.append(heuristic_quality(result["answer"], question["expected_contains"]))
        compactions += agent.compaction_count(chat_thread) + agent.compaction_count(recall_thread)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        memory_growth_bytes=sum(memory_size(user_id) for user_id in users) - initial_size,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render the required six metrics alongside agent names."""
    return tabulate(
        [[row.agent_name, row.agent_tokens_only, row.prompt_tokens_processed,
          f"{row.recall_score:.1%}", f"{row.response_quality:.3f}",
          row.memory_growth_bytes, row.compactions] for row in rows],
        headers=["Agent", "Agent tokens only", "Prompt tokens processed",
                 "Cross-session recall", "Response quality", "Memory growth (bytes)",
                 "Compactions"],
        tablefmt="github", disable_numparse=True,
    )


def _suite_config(config: LabConfig, name: str, conversations: list[dict[str, Any]]) -> LabConfig:
    """Reset only the dataset users' User.md files in this benchmark namespace."""
    state_dir = config.state_dir / "benchmarks" / name
    suite_config = replace(config, state_dir=state_dir)
    profiles = UserProfileStore(state_dir / "profiles")
    for user_id in {conversation["user_id"] for conversation in conversations}:
        profiles.path_for(user_id).unlink(missing_ok=True)
    return suite_config


def main() -> None:
    """Run both suites offline, even when live credentials are configured."""
    config = load_config(Path(__file__).resolve().parent.parent)
    suites = [
        ("Standard Benchmark", "standard", load_conversations(config.data_dir / "conversations.json")),
        ("Long-Context Stress Benchmark", "long_context",
         load_conversations(config.data_dir / "advanced_long_context.json")),
    ]
    print("Offline benchmark: estimated tokens; quality = expected-fact coverage (0..1).")
    print("Token totals include chat and recall. Each conversation uses a fresh recall thread.")
    for title, name, conversations in suites:
        suite_config = _suite_config(config, name, conversations)
        rows = [run_agent_benchmark(label, agent_type(suite_config, force_offline=True),
                                    conversations, suite_config)
                for label, agent_type in (("Baseline", BaselineAgent), ("Advanced", AdvancedAgent))]
        print(f"\n## {title}\n")
        print(format_rows(rows))


if __name__ == "__main__":
    main()
