"""Базовые типы инструментов."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class ToolKind(str, Enum):
    """Тип инструмента определяет, как с ним обращается policy и бюджет."""

    READ = "read"        # только чтение, без побочных эффектов
    WRITE = "write"      # создаёт или меняет файлы
    DELETE = "delete"    # удаляет файлы
    BASH = "bash"        # запускает команду
    CONTROL = "control"  # служебные инструменты runtime (submit_plan, finish_task)


# Действия с побочным эффектом: требуют подтверждения в safe, считаются
# в бюджете side_effects и записываются как started до выполнения.
SIDE_EFFECT_KINDS = {ToolKind.WRITE.value, ToolKind.DELETE.value, ToolKind.BASH.value}


class ToolError(Exception):
    """Ожидаемая ошибка инструмента: текст уйдёт модели как результат."""


@dataclass
class ToolResult:
    ok: bool
    data: dict


@dataclass
class ToolContext:
    """То, что runtime передаёт инструменту при выполнении."""

    workspace: Path
    policy: Any           # agent.policy.Policy (повторные проверки путей и команд)
    max_output: int       # сколько байт результата ещё можно вернуть (остаток бюджета)
    bash_timeout: float
    runtime: Any          # Agent - нужен только служебным инструментам
    run_id: str
    task_id: str | None
    call_id: str


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict            # JSON Schema аргументов (уходит модели)
    kind: ToolKind
    retryable: bool             # можно ли повторить вызов, ставший неопределённым после сбоя
    handler: Callable[[dict, ToolContext], ToolResult]
    path_args: tuple = ()       # имена аргументов-путей, которые проверяет policy
    phases: tuple = ("planning", "execution")
    requires_tasks: bool = False  # доступен только при исполнении плана с задачами

    def spec(self) -> dict:
        """Описание инструмента в формате OpenAI Chat Completions."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
