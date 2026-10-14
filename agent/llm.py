import copy
import json
import os
from pathlib import Path


class LLMError(Exception):
    """Ошибка обращения к модели (сеть, ключ, неверный ответ)."""


class OpenAIChatClient:
    def __init__(self, base_url=None, api_key=None, model=None, timeout: float = 180.0):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("пакет openai не установлен: pip install -r requirements.txt") from exc
        self.model = model or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise LLMError("не задана переменная окружения OPENAI_MODEL (см. .env.example)")
        self.client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "not-set",
            timeout=timeout,
        )

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        kwargs = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message
        except Exception as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        tool_calls = [
            {
                "id": tc.id or "",
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
            }
            for tc in (message.tool_calls or [])
        ]
        return {"content": message.content or "", "tool_calls": tool_calls}


class ScriptedLLM:
    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.requests: list[dict] = []

    @classmethod
    def from_file(cls, path) -> "ScriptedLLM":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        index = sum(1 for m in messages if m["role"] == "assistant")
        self.requests.append({
            "index": index,
            "tools": [t["function"]["name"] for t in tools],
            "messages": copy.deepcopy(messages),
        })
        if index >= len(self.responses):
            return {"content": "(сценарий закончился)", "tool_calls": []}
        return copy.deepcopy(self.responses[index])