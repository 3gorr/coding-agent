"""Реестр инструментов: поиск по имени, JSON Schema, безопасный вызов."""
from jsonschema import Draft7Validator

from ..policy import PolicyViolation
from .base import Tool, ToolContext, ToolError, ToolResult


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}
        self._validators = {t.name: Draft7Validator(t.parameters) for t in tools}

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self, phase: str, with_tasks: bool) -> list[dict]:
        """Инструменты, которые показываем модели в текущей фазе."""
        return [
            t.spec() for t in self._tools.values()
            if phase in t.phases and (with_tasks or not t.requires_tasks)
        ]

    def validate(self, tool: Tool, args) -> str | None:
        """None, если аргументы подходят под JSON Schema, иначе текст ошибки."""
        if not isinstance(args, dict):
            return "аргументы должны быть JSON-объектом"
        errors = sorted(self._validators[tool.name].iter_errors(args), key=lambda e: list(e.path))
        if not errors:
            return None
        first = errors[0]
        where = "/".join(str(p) for p in first.path) or "(корень)"
        return f"аргументы не соответствуют JSON Schema инструмента {tool.name}: {where}: {first.message}"

    def invoke(self, tool: Tool, args: dict, ctx: ToolContext) -> ToolResult:
        """Выполняет инструмент. Любая ошибка превращается в результат, а не роняет процесс."""
        try:
            return tool.handler(args, ctx)
        except ToolError as exc:
            return ToolResult(False, {"error": str(exc)})
        except PolicyViolation as exc:
            return ToolResult(False, {"error": f"policy: {exc}"})
        except Exception as exc:  # noqa: BLE001 - любая неожиданная ошибка инструмента
            return ToolResult(False, {"error": f"{type(exc).__name__}: {exc}"})
