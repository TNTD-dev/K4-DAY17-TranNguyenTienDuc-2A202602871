"""Shared deterministic responses for the lab's controlled memory comparison."""

import re


BASE_SYSTEM_PROMPT = (
    "Bạn là trợ lý trả lời bằng tiếng Việt. Chỉ dùng thông tin trong ngữ cảnh được cung cấp; "
    "nếu thiếu thông tin về người dùng, hãy nói rõ thay vì đoán."
)

_FACT_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "favorite_drink": "Đồ uống yêu thích",
    "favorite_food": "Món ăn yêu thích",
    "pet": "Thú nuôi",
    "pet_name": "Tên thú nuôi",
    "interests": "Mối quan tâm kỹ thuật",
    "response_style": "Phong cách trả lời",
}


def merge_profile_updates(facts: dict[str, str], updates: dict[str, str]) -> dict[str, str]:
    """Latest fields win; partial style reaffirmations retain unrelated constraints."""
    merged = {**facts, **updates}
    if "response_style" in updates:
        def slot(part):
            if part in ("ngắn gọn", "chi tiết"):
                return "length"
            if "bullet" in part:
                return "bullets"
            if "ví dụ" in part:
                return "examples"
            if "trade-off" in part:
                return "focus"
            return part

        parts = {}
        for style in (facts.get("response_style", ""), updates["response_style"]):
            for part in filter(None, (part.strip() for part in style.split(";"))):
                key = slot(part)
                if part == "có bullet" and re.search(r"\d+ bullet", parts.get(key, "")):
                    continue
                parts[key] = part
        merged["response_style"] = "; ".join(parts.values())
    return merged


def _format_parts(parts: list[str], style: str) -> str:
    count = re.search(r"\b([1-6])\s+bullet", style)
    if count:
        size = int(count[1])
        if len(parts) < size:
            notes = [
                "Nội dung được dùng trong ngữ cảnh hiện tại",
                "Bạn có thể đính chính nếu thông tin đã thay đổi",
                "Thông tin chưa được cung cấp sẽ không được suy đoán",
                "Các câu trả lời dựa trên thông tin đã được cung cấp",
                "Bạn có thể bổ sung thông tin khi cần",
            ]
            parts = parts + notes[:size - len(parts)]
        groups = [parts[i * len(parts) // size:(i + 1) * len(parts) // size] for i in range(size)]
        return "\n".join("- " + "; ".join(group).rstrip(".") + "." for group in groups)
    if "có bullet" in style:
        return "\n".join(f"- {part.rstrip('.')}." for part in parts)
    return "; ".join(parts).rstrip(".") + "."


def answer_from_facts(message: str, facts: dict[str, str]) -> str:
    """Render requested facts; never copy answers from question text or a dataset."""
    query = message.lower()
    recall = "?" in query or re.search(
        r"nhắc lại|nhớ lại|tóm tắt.*mình|mô tả.*mình|tên gì|ở đâu|nghề gì", query
    )
    style = facts.get("response_style", "")
    if not recall:
        return _format_parts(["Đã ghi nhận nội dung trong ngữ cảnh hiện tại"], style)
    patterns = {
        "name": r"\btên\b|là ai|biết.*không",
        "location": r"nơi ở|ở đâu|đang ở|còn ở",
        "profession": r"nghề|công việc",
        "favorite_drink": r"đồ uống",
        "favorite_food": r"món ăn",
        "pet": r"nuôi|thú cưng|thú nuôi",
        "interests": r"mối quan tâm|quan tâm chính|kỹ thuật chính",
        "response_style": r"style|phong cách|kiểu trả lời|trả lời.*thích",
    }
    if re.search(r"tóm tắt.*mình|mô tả.*mình", query):
        requested = list(patterns)
    else:
        requested = [key for key, pattern in patterns.items() if re.search(pattern, query)]
    if not requested:
        return _format_parts([
            "Mình chỉ có thể nhắc lại thông tin đã được cung cấp trong ngữ cảnh hiện tại"
        ], style)
    parts = []
    for key in requested:
        value = facts.get(key)
        if value:
            if key == "pet" and facts.get("pet_name"):
                value += f" tên {facts['pet_name']}"
            parts.append(f"{_FACT_LABELS[key]}: {value}")
        else:
            parts.append(f"{_FACT_LABELS[key]}: chưa có thông tin trong ngữ cảnh hiện tại")
    return _format_parts(parts, style)
