"""The correction ablation changes only its intended policy and stays isolated."""

import json

from benchmark_conflicts import run_conflict_comparison
from config import LabConfig


def test_conflict_ablation_is_scoped_repeatable_and_offline(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    conversations = [{
        "id": "correction", "user_id": "user",
        "turns": [
            "Mình tên là An. Mình ở Huế. Mình đang làm backend engineer.",
            "Mình tên là Bình. Mình hiện ở Đà Nẵng. Mình đang làm MLOps engineer.",
        ],
        "recall_questions": [{
            "question": "Nhắc lại tên, nơi ở và nghề nghiệp hiện tại của mình.",
            "expected_contains": ["Bình", "Đà Nẵng", "MLOps engineer"],
        }],
    }]
    for filename in ("conversations.json", "advanced_long_context.json"):
        (data_dir / filename).write_text(json.dumps(conversations, ensure_ascii=False), encoding="utf-8")
    inputs_before = {path: path.read_bytes() for path in data_dir.iterdir()}
    config = LabConfig(data_dir=data_dir, state_dir=tmp_path / "state")
    existing = config.state_dir / "profiles" / "user" / "User.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("existing user profile", encoding="utf-8")

    def forbid_live_model(*args, **kwargs):
        raise AssertionError("Correction comparison must stay offline")

    monkeypatch.setattr("agent_advanced.build_chat_model", forbid_live_model)
    evidence = run_conflict_comparison(config)
    assert run_conflict_comparison(config) == evidence
    for variants in evidence["datasets"].values():
        control, corrected = variants
        assert control["metrics"]["recall_score"] == 0.5
        assert corrected["metrics"]["recall_score"] == 1
        assert control["recall_answers"][0]["missing"] == ["Đà Nẵng", "MLOps engineer"]
        assert corrected["recall_answers"][0]["missing"] == []
        # Name remains updateable in both; only location/profession differ.
        for variant in variants:
            assert "- name: Bình" in variant["final_profiles"]["user"]
        assert "- location: Huế" in control["final_profiles"]["user"]
        profile = corrected["final_profiles"]["user"]
        assert "- location: Đà Nẵng" in profile and "- profession: MLOps engineer" in profile
        assert "Huế" not in profile and "backend engineer" not in profile
    assert existing.read_text(encoding="utf-8") == "existing user profile"
    assert {path: path.read_bytes() for path in data_dir.iterdir()} == inputs_before
