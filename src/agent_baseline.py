from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model, normalize_provider
from offline_response import BASE_SYSTEM_PROMPT, answer_from_facts, merge_profile_updates


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Thread-local history only: no persistent profile and no compaction."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self._thread_users: dict[str, str] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return answer and this turn's agent_tokens/prompt_tokens.

        Cumulative counters are available through token_usage/prompt_token_usage.
        A thread ID belongs to one user so accidental reuse cannot leak history.
        """
        if not user_id.strip() or not thread_id.strip():
            raise ValueError("user_id and thread_id must not be empty.")
        owner = self._thread_users.get(thread_id)
        if owner is not None and owner != user_id:
            raise ValueError("thread_id already belongs to a different user.")
        self._thread_users[thread_id] = user_id
        if self.force_offline or self.langchain_agent is None:
            return self._reply_offline(thread_id, message)
        return self._reply_live(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        return estimate_tokens(BASE_SYSTEM_PROMPT) + sum(
            estimate_tokens(item["content"]) for item in messages
        )

    def _record_turn(
        self, session: SessionState, message: str, answer: str,
        agent_tokens: int, prompt_tokens: int,
    ) -> dict[str, Any]:
        session.messages.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ])
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {"answer": answer, "agent_tokens": agent_tokens, "prompt_tokens": prompt_tokens}

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        prompt_messages = session.messages + [{"role": "user", "content": message}]
        facts = {}
        for item in prompt_messages:
            if item["role"] == "user":
                facts = merge_profile_updates(facts, extract_profile_updates(item["content"]))
        answer = answer_from_facts(message, facts)
        return self._record_turn(
            session, message, answer, estimate_tokens(answer), self._prompt_tokens(prompt_messages)
        )

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        response = result["messages"][-1]
        if response.type != "ai":
            raise RuntimeError("Live agent did not return an assistant message.")
        answer = response.text
        usage = response.usage_metadata or {}
        prompt_messages = session.messages + [{"role": "user", "content": message}]
        return self._record_turn(
            session, message, answer,
            usage.get("output_tokens", estimate_tokens(answer)),
            usage.get("input_tokens", self._prompt_tokens(prompt_messages)),
        )

    def _maybe_build_langchain_agent(self):
        """Use a RAM checkpointer only when live settings are available."""
        if self.force_offline:
            return None
        model_config = self.config.model
        provider = normalize_provider(model_config.provider)
        if provider == "custom":
            if not model_config.base_url:
                return None
        elif provider != "ollama" and not model_config.api_key:
            return None
        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        return create_agent(
            model=build_chat_model(model_config),
            tools=[],
            system_prompt=BASE_SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )
