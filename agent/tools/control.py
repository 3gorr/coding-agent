"""Служебные инструменты runtime.

submit_plan  - модель сдаёт структурированный план (только фаза planning);
finish_task  - модель заявляет, что текущая задача выполнена. Заявления мало:
               runtime сам проверяет verification задачи и только тогда
               помечает её done.
Логика живёт в Agent (ctx.runtime), здесь только описания и схемы.
"""
from ..plan import PLAN_SCHEMA
from .base import Tool, ToolContext, ToolKind, ToolResult


def _submit_plan(args: dict, ctx: ToolContext) -> ToolResult:
    return ctx.runtime.submit_plan(args)


def _finish_task(args: dict, ctx: ToolContext) -> ToolResult:
    return ctx.runtime.finish_task(args["task_id"], args.get("summary", ""))


SUBMIT_PLAN = Tool(
    name="submit_plan",
    description="Сдать план работы на утверждение пользователю. Вызывается один раз в конце "
                "фазы планирования. Каждый шаг станет задачей с проверкой результата.",
    parameters=PLAN_SCHEMA,
    kind=ToolKind.CONTROL,
    retryable=True,  # повтор перезаписывает тот же план, побочных эффектов в workspace нет
    handler=_submit_plan,
    phases=("planning",),
)

FINISH_TASK = Tool(
    name="finish_task",
    description="Сообщить, что текущая задача выполнена. Runtime проверит результат задачи "
                "и только после успешной проверки отметит её выполненной.",
    parameters={
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "minLength": 1},
            "summary": {"type": "string", "description": "Что сделано"},
        },
        "required": ["task_id"],
        "additionalProperties": False,
    },
    kind=ToolKind.CONTROL,
    retryable=True,  # для уже завершённой задачи повтор возвращает сохранённое подтверждение
    handler=_finish_task,
    phases=("execution",),
    requires_tasks=True,
)

CONTROL_TOOLS = [SUBMIT_PLAN, FINISH_TASK]
