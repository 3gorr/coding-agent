import json

from .budget import charge_call_start, settle_output
from .store import now

TRUNC_MARK = "\n...[ОБРЕЗАНО RUNTIME: показано {shown} из {total} байт, вывод неполный]"


def render_payload(payload: dict, limit: int) -> tuple[str, bool]:
    """JSON-строка результата не длиннее limit байт; при обрезке явная пометка."""
    text = json.dumps(payload, ensure_ascii=False)
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text, False
    longest_mark = TRUNC_MARK.format(shown=len(data), total=len(data)).encode("utf-8")
    shown = max(0, limit - len(longest_mark))
    content = data[:shown].decode("utf-8", errors="ignore") + TRUNC_MARK.format(shown=shown, total=len(data))
    encoded = content.encode("utf-8")
    if len(encoded) > limit:
        content = encoded[:limit].decode("utf-8", errors="ignore")
    return content, True


class Executor:
    def __init__(self, store, registry, crash):
        self.store = store
        self.registry = registry
        self.crash = crash

    def execute(self, run_id: str, row: dict, tool, make_ctx) -> dict:
        """Выполнить вызов. BudgetExceeded бросается ДО начала выполнения."""
        key = {"run_id": run_id, "call_id": row["call_id"], "attempt": row["attempt"]}

        with self.store.tx():
            run = self.store.get_run(run_id)
            used, reserve = charge_call_start(run, tool.kind.value)
            task_id = run["current_task"] if run["phase"] == "execution" else None
            self.store.update_run(run_id, budget_used=used)
            self.store.update_call(**key, status="started", output_reserved=reserve,
                                   task_id=task_id, started_at=now())

        self.crash("after_start")
        result = self.registry.invoke(tool, row["args"], make_ctx(reserve, task_id, row["call_id"]))
        payload = {"ok": bool(result.ok), **result.data}
        content, truncated = render_payload(payload, reserve)
        if truncated:
            payload["truncated_by_runtime"] = True
        self.crash("after_exec")

        charged = reserve if truncated else len(content.encode("utf-8"))
        with self.store.tx():
            run = self.store.get_run(run_id)
            self.store.update_run(run_id, budget_used=settle_output(run, reserve, charged))
            self.store.update_call(
                **key, status="done", result=payload, content=content,
                truncated=int(truncated), done_seq=self.store.next_done_seq(run_id),
                finished_at=now(),
            )
            self.store.append_message(
                run_id, {"role": "tool", "tool_call_id": row["call_id"], "content": content},
                phase=run["phase"],
            )
        return payload
