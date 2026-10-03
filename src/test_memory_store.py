"""Step 4 checks, isolated from the unfinished agent integration tests."""

import json
from pathlib import Path

import pytest

from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    summarize_messages,
)


def test_token_estimator_empty_deterministic_and_monotonic():
    assert estimate_tokens("") == estimate_tokens(" \n ") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("xin chào") == estimate_tokens("xin chào")
    counts = [estimate_tokens("a" * length) for length in range(100)]
    assert counts == sorted(counts)


def test_profile_read_write_edit_and_utf8_bytes(tmp_path):
    store = UserProfileStore(tmp_path)
    assert store.read_text("an") == ""
    assert store.file_size("an") == 0
    content = "# Hồ sơ\nHuế, Huế\n"
    path = store.write_text("an", content)
    assert path == tmp_path / "an" / "User.md"
    assert store.file_size("an") == len(content.encode("utf-8"))
    assert store.read_text("an") == content
    assert store.edit_text("an", "Huế", "Đà Nẵng")
    assert store.read_text("an") == "# Hồ sơ\nĐà Nẵng, Huế\n"
    assert not store.edit_text("an", "missing", "new")
    assert not store.edit_text("an", "", "new")
    assert not store.edit_text("missing-user", "missing", "new")


def test_profile_reopened_store_and_user_isolation(tmp_path):
    store = UserProfileStore(tmp_path)
    store.upsert_fact("a", "name", "An")
    store.upsert_fact("b", "name", "Bình")
    reopened = UserProfileStore(tmp_path)
    assert reopened.facts("a") == {"name": "An"}
    assert reopened.facts("b") == {"name": "Bình"}


def test_correction_deduplicates_and_preserves_notes(tmp_path):
    store = UserProfileStore(tmp_path)
    store.write_text("an", "# Profile\n- location: Huế\n- location: Huế\nNotes remain.\n")
    assert store.upsert_facts("an", {"location": "Đà Nẵng", "name": "An"})
    assert store.facts("an") == {"location": "Đà Nẵng", "name": "An"}
    text = store.read_text("an")
    assert "Huế" not in text
    assert text.count("- location:") == 1
    assert "Notes remain." in text
    assert not store.upsert_fact("an", "location", "Đà Nẵng")
    assert not store.upsert_facts("an", {})


@pytest.mark.parametrize("user_id", ["../../escape", "/absolute", "a/b", "a b", "Đức", ".", "a" * 200])
def test_paths_stay_inside_root_and_are_deterministic(tmp_path, user_id):
    store = UserProfileStore(tmp_path)
    path = store.path_for(user_id)
    assert path.is_relative_to(tmp_path)
    assert path == store.path_for(user_id)
    # Even an ID matching an encoded directory cannot alias its owner.
    assert store.path_for(path.parent.name) != path


def test_ids_with_similar_slugs_do_not_collide(tmp_path):
    store = UserProfileStore(tmp_path)
    assert store.path_for("a/b") != store.path_for("a b")
    assert store.path_for("Đức") != store.path_for("Duc")
    with pytest.raises(ValueError):
        store.path_for(" ")


def test_symlink_escape_is_rejected(tmp_path):
    root, outside = tmp_path / "profiles", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "an").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="inside"):
        UserProfileStore(root).write_text("an", "profile")
    assert not (outside / "User.md").exists()


@pytest.mark.parametrize("key,value", [("bad key", "ok"), ("name", ""), ("name", "An\n- location: Huế")])
def test_invalid_fact_cannot_inject_markdown(tmp_path, key, value):
    store = UserProfileStore(tmp_path)
    with pytest.raises(ValueError):
        store.upsert_fact("an", key, value)
    assert store.read_text("an") == ""


def test_extract_basic_profile_and_style():
    message = (
        "Mình tên là An, hiện ở Hải Phòng và đang làm data engineer cho một công ty. "
        "Đồ uống yêu thích là trà xanh. Món ăn yêu thích là phở. "
        "Mình nuôi một con mèo tên Mun. Mình thích Python và RAG. "
        "Mình muốn bạn trả lời ngắn gọn thành 5 bullet có ví dụ thực tế và nhấn trade-off."
    )
    facts = extract_profile_updates(message)
    assert facts["name"] == "An"
    assert facts["location"] == "Hải Phòng"
    assert facts["profession"] == "data engineer"
    assert facts["favorite_drink"] == "trà xanh"
    assert facts["favorite_food"] == "phở"
    assert facts["pet"] == "mèo"
    assert facts["pet_name"] == "Mun"
    assert "Python" in facts["interests"] and "RAG" in facts["interests"]
    assert "5 bullet" in facts["response_style"] and "trade-off" in facts["response_style"]


def test_extract_latest_explicit_correction():
    facts = extract_profile_updates(
        "Mình ở Huế, giờ mình ở Đà Nẵng. "
        "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer."
    )
    assert facts == {"location": "Đà Nẵng", "profession": "MLOps engineer"}


@pytest.mark.parametrize("message,expected", [
    ("Mình ở Đà Nẵng, trước đây mình ở Huế.", {"location": "Đà Nẵng"}),
    ("Trước đây mình ở Huế, nhưng mình hiện ở Đà Nẵng.", {"location": "Đà Nẵng"}),
    ("Hồi trước mình đang làm backend engineer.", {}),
    ("Bạn tôi tên là An.", {}),
    ("Bạn của mình ở Huế.", {}),
    ("Chị gái tôi đang làm data engineer.", {}),
    ("Bạn tôi tên là An, hiện ở Huế và đang làm data engineer.", {}),
    ("Bạn tôi thích Python.", {}),
])
def test_historical_and_third_party_assertions_do_not_override_self(message, expected):
    assert extract_profile_updates(message) == expected


@pytest.mark.parametrize("message", [
    "Mình tên gì? Hiện tại mình ở đâu?",
    "Mình tên là ai",
    "Nhắc lại giúp mình: tên, món ăn yêu thích và mình nuôi con gì.",
    "Nhắc lại giúp mình tên và style trả lời mình thích.",
    "Nếu sau này mình ở Hà Nội thì sao?",
    "Mình đùa rằng giờ chuyển sang product manager.",
    "Hà Nội chỉ là nơi mình vừa bay ra họp chứ không phải nơi ở hiện tại.",
    'Một email ghi: "Mình ở Hà Nội".',
    "Thông tin ổn định gồm tên, nơi ở, đồ uống yêu thích và style trả lời.",
    "Hôm nay mình làm việc ở quán cà phê.",
    "Mình ở Hà Nội để họp hai ngày.",
    "Nghe nói mình tên là An.",
    "Tạm thời mình muốn trả lời thành 8 bullet.",
])
def test_questions_noise_and_temporary_preferences_are_not_facts(message):
    assert extract_profile_updates(message) == {}


def test_mixed_assertion_and_question_keeps_only_assertion():
    assert extract_profile_updates("Mình tên là An. Bạn nhớ mình ở đâu không?") == {"name": "An"}


def test_style_assertion_can_mention_summarizing_without_becoming_a_recall_request():
    facts = extract_profile_updates(
        "Mình muốn bạn trả lời ngắn gọn thành 3 bullet, ưu tiên trade-off hơn là tóm tắt chung chung."
    )
    assert facts["response_style"] == "ngắn gọn; 3 bullet; ưu tiên trade-off"


def test_summary_is_bounded_and_merges_corrections():
    first = summarize_messages([
        {"role": "user", "content": "Mình ở Huế. Quyết định dùng PostgreSQL."},
        {"role": "assistant", "content": "Đã ghi nhận. " * 100},
    ])
    second = summarize_messages([
        {"role": "summary", "content": first},
        {"role": "user", "content": "Mình ở Đà Nẵng."},
    ])
    assert "- location: Đà Nẵng" in second
    assert "- location: Huế" not in second
    assert "Quyết định dùng PostgreSQL" in second
    assert len(second.splitlines()) <= 6
    assert summarize_messages([]) == ""
    assert summarize_messages([], max_items=0) == ""


def test_compact_trigger_exact_tail_and_thread_isolation():
    manager = CompactMemoryManager(threshold_tokens=80, keep_messages=2)
    messages = [{"role": "user", "content": f"Message {i}: " + "nội dung " * 20} for i in range(5)]
    for message in messages:
        manager.append("thread-a", **message)
    context = manager.context("thread-a")
    assert manager.compaction_count("thread-a") > 0
    assert context["messages"] == messages[-2:]
    assert context["summary"]
    assert manager.context("thread-b") == {"messages": [], "summary": "", "compactions": 0}
    context["messages"][0]["content"] = "changed externally"
    assert manager.context("thread-a")["messages"] == messages[-2:]


def test_below_threshold_does_not_compact():
    manager = CompactMemoryManager(100, 2)
    for _ in range(5):
        manager.append("thread", "user", "short")
    assert manager.compaction_count("thread") == 0
    assert len(manager.context("thread")["messages"]) == 5


def test_repeated_compaction_keeps_summary_bounded_and_reduces_total_prompt_load():
    manager = CompactMemoryManager(100, 2)
    full_history = []
    baseline_load = compact_load = 0
    for i in range(60):
        content = f"Lượt {i}: " + "Thông tin dài về công việc. " * 12
        full_history.append(content)
        manager.append("thread", "user", content)
        context = manager.context("thread")
        assert estimate_tokens(context["summary"]) <= 25
        baseline_load += sum(map(estimate_tokens, full_history))
        compact_load += estimate_tokens(context["summary"]) + sum(
            estimate_tokens(m["content"]) for m in context["messages"]
        )
    assert manager.compaction_count("thread") > 1
    assert compact_load < baseline_load


def test_existing_summary_is_included_in_trigger():
    manager = CompactMemoryManager(50, 2)
    manager.state["thread"] = {
        "messages": [{"role": "user", "content": "a" * 80}] * 2,
        "summary": "s" * 80,
        "compactions": 1,
    }
    manager.append("thread", "user", "a" * 4)
    assert manager.compaction_count("thread") == 2


def test_oversized_recent_input_is_preserved():
    manager = CompactMemoryManager(20, 2)
    manager.append("thread", "user", "a" * 1000)
    assert manager.context("thread")["messages"][0]["content"] == "a" * 1000
    assert manager.compaction_count("thread") == 0


@pytest.mark.parametrize("threshold,keep", [(0, 2), (10, 0), (-1, 2)])
def test_invalid_compact_configuration(threshold, keep):
    with pytest.raises(ValueError):
        CompactMemoryManager(threshold, keep)


def test_dataset_profiles_and_stress_compaction(tmp_path):
    store = UserProfileStore(tmp_path)
    manager = CompactMemoryManager(1000, 4)
    root = Path(__file__).resolve().parent.parent
    for filename in ("conversations.json", "advanced_long_context.json"):
        for conversation in json.loads((root / "data" / filename).read_text(encoding="utf-8")):
            for message in conversation["turns"]:
                store.upsert_facts(conversation["user_id"], extract_profile_updates(message))
                manager.append(conversation["id"], "user", message)
                manager.append(conversation["id"], "assistant", "Đã ghi nhận.")
    standard = store.facts("dungct")
    assert standard["location"] == "Huế"
    assert standard["profession"] == "MLOps engineer"
    assert standard["favorite_drink"] == "cà phê sữa đá"
    assert standard["pet"] == "corgi"
    stress = store.facts("dungct_stress")
    assert stress["name"] == "DũngCT Stress"
    assert stress["location"] == "Đà Nẵng"
    assert stress["profession"] == "MLOps engineer"
    assert "3 bullet" in stress["response_style"]
    assert manager.compaction_count("stress-01") > 1
    assert store.file_size("dungct_stress") > 0
