from __future__ import annotations

import hashlib
import os
import re
import tempfile
import textwrap
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict


def estimate_tokens(text: str) -> int:
    """Deterministic character-based estimate, shared by memory and benchmark."""
    stripped = (text or "").strip()
    return max(1, len(stripped) // 4) if stripped else 0


_FACT_LINE = re.compile(r"^- ([a-z][a-z0-9_]*): (.+)$")


@dataclass
class UserProfileStore:
    """UTF-8 profiles, one User.md per user, with atomic file replacement."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        if not user_id.strip():
            raise ValueError("user_id must not be empty.")
        if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", user_id):
            directory = user_id
        else:
            normalized = unicodedata.normalize("NFKD", user_id)
            slug = re.sub(r"[^a-z0-9_-]+", "-", normalized.encode("ascii", "ignore").decode().lower())
            digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
            # '~' separates encoded IDs from the namespace of already-safe IDs.
            directory = f"~{slug.strip('-')[:40] or 'user'}-{digest}"
        root = self.root_dir.resolve()
        path = (root / directory / "User.md").resolve()
        if not path.is_relative_to(root):
            raise ValueError("Profile path must remain inside root_dir.")
        return path

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if not search_text or search_text not in content:
            return False
        updated = content.replace(search_text, replacement, 1)
        if updated == content:
            return False
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return {
            match[1]: match[2]
            for line in self.read_text(user_id).splitlines()
            if (match := _FACT_LINE.fullmatch(line))
        }

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        return self.upsert_facts(user_id, {key: value})

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> bool:
        """Replace affected fact lines once, preserving unrelated Markdown notes."""
        if not updates:
            return False
        for key, value in updates.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
                raise ValueError("Fact keys must be lowercase identifiers.")
            if not value.strip() or "\n" in value or "\r" in value:
                raise ValueError("Fact values must be nonempty single-line text.")
        current = self.read_text(user_id)
        lines = (current or "# User Profile\n").splitlines()
        seen = set()
        updated = []
        for line in lines:
            match = _FACT_LINE.fullmatch(line)
            if match and match[1] in updates:
                key = match[1]
                if key not in seen:
                    updated.append(f"- {key}: {updates[key].strip()}")
                    seen.add(key)
            else:
                updated.append(line)
        for key in sorted(updates.keys() - seen):
            updated.append(f"- {key}: {updates[key].strip()}")
        content = "\n".join(updated).rstrip() + "\n"
        if content == current:
            return False
        self.write_text(user_id, content)
        return True


_USER = r"(?:mình|tôi|tớ)"
_STOP = re.compile(
    r",|\b(?:và\s+(?:đang|mình|tôi|ưu tiên)|chứ|nhưng|dù|cho|để|vì|"
    r"trong|vài|mỗi|như cũ|không đổi|nữa nhé)\b", re.IGNORECASE
)
_UNCERTAIN = re.compile(
    r"\b(?:đùa|giả sử|giả vờ|nếu sau này|hay là|có thể sẽ|"
    r"muốn chuyển|định chuyển|nghe nói|email|(?:tài liệu|bài báo|đồng nghiệp)\s+(?:nói|viết|ghi|bảo))\b"
    r"|\bví dụ\s*[:,]", re.IGNORECASE
)
_QUESTION = re.compile(
    r"\b(?:là gì|là ai|tên gì|ở đâu|nghề gì|con gì|thế nào|bao nhiêu|đúng không|phải không)\b",
    re.IGNORECASE,
)
_RECALL_REQUEST = re.compile(
    r"^(?:(?:bạn|hãy|thử|giúp|sau đó|ngoài ra)\s+)*(?:nhắc lại|nhớ lại|tóm tắt|mô tả)\b"
    r"|,\s*(?:nhắc lại|nhớ lại|tóm tắt|mô tả)\b", re.IGNORECASE,
)


def _fact_value(value: str) -> str:
    return _STOP.split(value, maxsplit=1)[0].strip(" :,-.!?\"'“”")


def _current_self_assertion(sentence: str, offset: int) -> bool:
    """Reject known historical clauses and possessive third-party subjects.

    This is a conservative grammar guard, not a general temporal/coreference parser.
    """
    prefix = re.split(r",|\b(?:nhưng|còn)\b", sentence[:offset], flags=re.IGNORECASE)[-1]
    if re.search(r"\b(?:trước đây|trước kia|hồi trước|hồi đó|ngày trước|lúc trước|trước đó)\b",
                 prefix, re.IGNORECASE):
        return False
    return not re.search(
        r"\b(?:bạn(?: thân)?|đồng nghiệp|anh(?: trai)?|chị(?: gái)?|em(?: trai| gái)?|"
        r"mẹ|bố|ba|vợ|chồng|người yêu)\s+(?:của\s+)?$", prefix, re.IGNORECASE,
    )


def extract_profile_updates(message: str) -> dict[str, str]:
    """Conservative Vietnamese assertion patterns, not a general entity parser.

    Questions, hypothetical/quoted claims and temporary requests are excluded.
    Matches are processed in order so the last explicit correction wins.
    """
    updates = {}
    patterns = {
        "name": rf"\b(?:{_USER}\s+tên(?:\s+là)?|tên\s+{_USER}(?:\s+là)?)\s+([^,.!?;]+)",
        "location": (
            rf"\b(?:{_USER}\s+(?:(?:hiện(?: tại)?|giờ|đang|vẫn)\s+)*"
            r"(?:ở|sống ở|làm việc ở)|nơi ở(?: hiện tại)?(?: của mình)?\s+(?:vẫn\s+)?là)\s+([^,.!?;]+)"
        ),
        "profession": (
            rf"\b(?:{_USER}\s+(?:(?:hiện(?: tại)?|đang|vẫn)\s+)*làm(?: nghề)?|"
            r"nghề(?: nghiệp)?(?: hiện tại)?(?: của mình)?\s+(?:(?:thì|vẫn)\s+)*là|"
            r"giờ\s+chuyển sang)\s+([^,.!?;]+)"
        ),
        "favorite_drink": rf"\bđồ uống yêu thích(?: của {_USER})?(?:\s+là\s+|\s*:\s*)([^,.!?;]+)",
        "favorite_food": rf"\bmón ăn yêu thích(?: của {_USER})?(?:\s+là\s+|\s*:\s*)([^,.!?;]+)",
        "pet": rf"\b{_USER}\s+nuôi\s+(?:một\s+)?(?:(?:bé|con)\s+)?([^,.!?;]+)",
    }
    for sentence_match in re.finditer(r"[^.!?;\n]+[.!?;\n]?", message):
        sentence = sentence_match[0].strip()
        if (
            sentence.endswith("?") or _QUESTION.search(sentence)
            or _RECALL_REQUEST.search(sentence) or _UNCERTAIN.search(sentence)
        ):
            continue
        if re.search(r"\b(?:tạm thời|chỉ hôm nay|riêng lượt này)\b", sentence, re.IGNORECASE):
            continue
        matches = []
        for key, pattern in patterns.items():
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                if _current_self_assertion(sentence, match.start()):
                    matches.append((match.start(), key, match[1]))
        has_self_subject = any(
            _current_self_assertion(sentence, match.start())
            for match in re.finditer(rf"\b{_USER}\b", sentence, re.IGNORECASE)
        )
        if has_self_subject:
            for match in re.finditer(r",\s*hiện(?: tại)?\s+ở\s+(.+)", sentence, re.IGNORECASE):
                matches.append((match.start(), "location", match[1]))
            for match in re.finditer(r"\bvà\s+đang\s+làm\s+(.+)", sentence, re.IGNORECASE):
                if _current_self_assertion(sentence, match.start()):
                    matches.append((match.start(), "profession", match[1]))
        for _, key, raw in sorted(matches):
            value = _fact_value(raw)
            if not value:
                continue
            if key == "profession" and re.match(r"việc\b", value, re.IGNORECASE):
                continue
            if key == "location" and (
                re.match(r"(?:quán|văn phòng|công ty|khách sạn|nhà hàng)\b", value, re.IGNORECASE)
                or re.match(r"(?:hôm nay|ngày mai|chiều nay|sáng nay)\b", sentence, re.IGNORECASE)
                or re.search(
                    r"\b(?:để (?:đi )?họp|đi du lịch|chỉ ghé|chỉ đến|không phải nơi ở)\b",
                    sentence, re.IGNORECASE,
                )
            ):
                continue
            if key == "pet":
                pet = re.split(r"\s+tên\s+", value, maxsplit=1, flags=re.IGNORECASE)
                updates[key] = pet[0]
                if len(pet) == 2:
                    updates["pet_name"] = pet[1]
            else:
                updates[key] = value
        interest = re.search(
            rf"\b{_USER}\s+(?:(?:vẫn|rất|đang)\s+)*(?:thích|quan tâm(?: nhiều)?(?:\s+đến)?)\s+(.+)",
            sentence, re.IGNORECASE,
        )
        if interest and _current_self_assertion(sentence, interest.start()):
            terms = re.findall(
                r"\b(?:Python|AI(?: ứng dụng| agent)?|MLOps|RAG|LangChain|LangGraph)\b",
                interest[1], re.IGNORECASE,
            )
            if terms:
                updates["interests"] = ", ".join(dict.fromkeys(terms))
        if re.search(r"trả lời|giải thích|style", sentence, re.IGNORECASE) and re.search(
            rf"{_USER}.*(?:muốn|thích)|hãy trả lời|style trả lời.*(?:giữ|là)",
            sentence, re.IGNORECASE,
        ):
            style = []
            if re.search(r"chi tiết|trả lời\s+dài", sentence, re.IGNORECASE):
                style.append("chi tiết")
            elif re.search(r"ngắn|gọn", sentence, re.IGNORECASE):
                style.append("ngắn gọn")
            bullets = re.search(r"\b(\d+)\s+bullet", sentence, re.IGNORECASE)
            if re.search(r"(?:không|bỏ)\s+(?:dùng\s+)?bullet", sentence, re.IGNORECASE):
                style.append("không dùng bullet")
            elif bullets:
                style.append(f"{bullets[1]} bullet")
            elif re.search(r"\bbullet\b", sentence, re.IGNORECASE):
                style.append("có bullet")
            if re.search(r"(?:không|bỏ)\s+(?:dùng\s+)?ví dụ", sentence, re.IGNORECASE):
                style.append("không dùng ví dụ")
            elif re.search(r"ví dụ thực (?:tế|chiến)", sentence, re.IGNORECASE):
                style.append("có ví dụ thực tế/thực chiến")
            if "trade-off" in sentence.lower():
                style.append("ưu tiên trade-off")
            if style:
                updates["response_style"] = "; ".join(style)
    return updates


def _shorten(text: str, width: int) -> str:
    if width < 1:
        return ""
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder="…")


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Bounded extractive summary: latest facts and short decision/context notes.

    Previous summaries are merged as structured lines, never nested transcripts.
    This heuristic can lose details; durable user facts belong in User.md.
    """
    if max_items < 1:
        return ""
    facts = {}
    notes = []
    for message in messages:
        role, content = message["role"], message["content"]
        if role == "summary":
            for line in content.splitlines():
                match = _FACT_LINE.fullmatch(line)
                if match and match[1] not in ("user", "assistant", "note"):
                    facts[match[1]] = match[2]
                elif line.strip():
                    notes.append(line)
        else:
            if role == "user":
                facts.update(extract_profile_updates(content))
            snippet = _shorten(content, 160)
            if snippet:
                notes.append(f"- {role}: {snippet}")
    # Reserve up to two slots for thread context beyond the user's profile.
    fact_items = [f"- {key}: {_shorten(value, 160)}" for key, value in facts.items()]
    fact_items = fact_items[:max(0, max_items - min(2, len(notes)))]
    unique_notes = list(dict.fromkeys(notes))
    ranked = sorted(enumerate(unique_notes), key=lambda item: (
        bool(re.search(r"quyết định|mục tiêu|deadline|TODO|blocker|không được", item[1], re.IGNORECASE)),
        item[0],
    ), reverse=True)
    selected = sorted(ranked[:max_items - len(fact_items)])
    return "\n".join(fact_items + [line for _, line in selected])


class ThreadMemory(TypedDict):
    messages: list[dict[str, str]]
    summary: str
    compactions: int


@dataclass
class CompactMemoryManager:
    """Per-thread history with a bounded summary and an exact recent-message tail."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, ThreadMemory] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens < 1 or self.keep_messages < 1:
            raise ValueError("threshold_tokens and keep_messages must be positive.")

    def append(self, thread_id: str, role: str, content: str) -> None:
        if role not in ("user", "assistant", "system", "tool"):
            raise ValueError("Unsupported message role.")
        thread = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        thread["messages"].append({"role": role, "content": content})
        total = estimate_tokens(thread["summary"]) + sum(
            estimate_tokens(message["content"]) for message in thread["messages"]
        )
        if total <= self.threshold_tokens or len(thread["messages"]) <= self.keep_messages:
            return
        older = thread["messages"][:-self.keep_messages]
        recent = thread["messages"][-self.keep_messages:]
        if thread["summary"]:
            older = [{"role": "summary", "content": thread["summary"]}] + older
        summary = summarize_messages(older)
        recent_tokens = sum(estimate_tokens(message["content"]) for message in recent)
        # Bound summary to 25% of the trigger budget and ensure each compact saves tokens.
        budget = min(max(1, self.threshold_tokens // 4), max(0, total - recent_tokens - 1))
        summary = _shorten_summary(summary, budget * 4)
        thread["messages"] = recent
        thread["summary"] = summary
        thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.get(thread_id, {"messages": [], "summary": "", "compactions": 0})
        return {
            "messages": [message.copy() for message in thread["messages"]],
            "summary": thread["summary"],
            "compactions": thread["compactions"],
        }

    def compaction_count(self, thread_id: str) -> int:
        return self.state.get(thread_id, {}).get("compactions", 0)


def _shorten_summary(summary: str, max_chars: int) -> str:
    """Keep line structure for the next merge and truncate only the final item."""
    lines = []
    remaining = max_chars
    for line in summary.splitlines():
        shortened = _shorten(line, remaining)
        if not shortened:
            break
        lines.append(shortened)
        remaining -= len(shortened) + 1
        if remaining <= 0:
            break
    return "\n".join(lines)
