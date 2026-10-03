"""Offline ablation of correction handling; production agents remain unchanged."""

from __future__ import annotations

import argparse
import json
import tempfile
import unicodedata
from dataclasses import asdict, replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from benchmark import BenchmarkRow, format_rows, load_conversations, run_agent_benchmark
from config import LabConfig, load_config
from memory_store import UserProfileStore, extract_profile_updates


class FirstValueWinsProfileStore(UserProfileStore):
    """Experimental control: ignore corrections to location/profession only.

    All other fields, extraction, compaction and response generation stay the same.
    This deliberately limited control is not an alternative production policy.
    """

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> bool:
        current = self.facts(user_id)
        accepted = {key: value for key, value in updates.items()
                    if key not in ("location", "profession") or key not in current}
        return super().upsert_facts(user_id, accepted)


def run_conflict_comparison(config: LabConfig) -> dict:
    results = {}
    with tempfile.TemporaryDirectory(prefix="memory-conflicts-") as temporary:
        for suite, filename in (("standard", "conversations.json"),
                                ("long_context", "advanced_long_context.json")):
            conversations = load_conversations(config.data_dir / filename)
            variants = []
            for label, first_value_wins in (("First value wins", True),
                                            ("Conflict handling", False)):
                variant = replace(config, state_dir=Path(temporary) / suite / label)
                agent = AdvancedAgent(variant, force_offline=True)
                if first_value_wins:
                    agent.profile_store = FirstValueWinsProfileStore(variant.state_dir / "profiles")
                answers = []
                recall_items = [(conversation, question) for conversation in conversations
                                for question in conversation["recall_questions"]]
                original_reply = agent.reply

                def traced_reply(user_id, thread_id, message):
                    reply = original_reply(user_id, thread_id, message)
                    if thread_id.startswith("recall:"):
                        conversation, question = recall_items[len(answers)]
                        normalized = unicodedata.normalize("NFC", reply["answer"]).casefold()
                        answers.append({
                            "conversation_id": conversation["id"], "question": message,
                            "answer": reply["answer"],
                            "expected_contains": question["expected_contains"],
                            "missing": [fact for fact in question["expected_contains"]
                                        if unicodedata.normalize("NFC", fact).casefold() not in normalized],
                        })
                    return reply

                agent.reply = traced_reply
                row = run_agent_benchmark(label, agent, conversations, variant)
                profiles = {user: agent.profile_store.read_text(user)
                            for user in sorted({item["user_id"] for item in conversations})}
                variants.append({"metrics": asdict(row), "recall_answers": answers,
                                 "final_profiles": profiles})
            results[suite] = variants
    probes = []
    for message, expected in (
        ("Mình ở Đà Nẵng, trước đây mình ở Huế.", {"location": "Đà Nẵng"}),
        ("Bạn tôi tên là An.", {}),
    ):
        actual = extract_profile_updates(message)
        probes.append({"message": message, "expected_updates": expected,
                       "actual_updates": actual, "matches_expected": actual == expected})
    return {
        "mode": "offline",
        "compact_threshold_tokens": config.compact_threshold_tokens,
        "compact_keep_messages": config.compact_keep_messages,
        "control": "First value wins for location/profession; all other behavior unchanged",
        "datasets": results,
        "extraction_limit_probes": probes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON evidence file")
    args = parser.parse_args()
    evidence = run_conflict_comparison(load_config(Path(__file__).resolve().parent.parent))
    for suite, variants in evidence["datasets"].items():
        print(f"\n## {suite}\n")
        print(format_rows([BenchmarkRow(**variant["metrics"]) for variant in variants]))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
