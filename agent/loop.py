import difflib
import json
import uuid
from pathlib import Path

from .budget import (BudgetExceeded, charge_model_request, check_batch, format_budget,
                     load_budget_config, remaining, zero_used)
from .crash import CrashPoints
from .executor import Executor
from .llm import LLMError
from .plan import render_plan, validate_plan
from .policy import Policy, PolicyViolation
from .prompts import PLANNING_REMINDER, build_system_prompt, task_reminder
from .snapshot import hash_tree
from .store import now
from .tasks import checklist, create_tasks, next_ready, start_task, task_brief, verify_task
from .tools import default_registry
from .tools.base import ToolContext, ToolResult

PAUSED = {"awaiting_approval", "awaiting_plan_approval", "awaiting_recovery", "model_error"}
TERMINAL = {"finished", "failed", "budget_exhausted"}
EXECUTABLE = {"allowed", "approved"}


class UserError(Exception):
    """Команда пользователя неприменима к текущему состоянию запуска."""


def _canonical(args, raw) -> str:
    if args is not None:
        return json.dumps(args, sort_keys=True, ensure_ascii=False)
    return raw or ""


def _short(value, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "..."


class Agent:
    def __init__(self, store, run_id: str, llm, *, registry=None, crash=None, out=print):
        self.store = store
        self.run_id = run_id
        self.llm = llm
        self.out = out
        run = store.get_run(run_id)
        self.workspace = Path(run["workspace"])
        self.policy = Policy(run["policy_path"], self.workspace, protected=[store.path])
        self.registry = registry or default_registry()
        self.crash = crash or CrashPoints()
        self.bash_timeout = float((run.get("settings") or {}).get("bash_timeout", 30))
        self.executor = Executor(store, self.registry, self.crash)


    @staticmethod
    def create_run(store, task: str, *, workspace, policy_path, mode: str = "safe",
                   use_plan: bool = True, budget: dict | None = None,
                   bash_timeout: float = 30.0) -> str:
        if mode not in ("safe", "auto"):
            raise UserError("режим должен быть safe или auto")
        policy = Policy(policy_path, workspace, protected=[store.path])
        phase = "planning" if use_plan else "execution"
        with store.tx():
            run_id = store.create_run(
                task=task, mode=mode, phase=phase, status="running",
                workspace=str(Path(workspace).resolve()),
                policy_path=str(Path(policy_path).resolve()),
                use_plan=int(use_plan), settings={"bash_timeout": bash_timeout},
                budget_limits=budget or load_budget_config(None), budget_used=zero_used(),
                start_hashes=hash_tree(workspace),
            )
            store.append_message(run_id, {"role": "system",
                                          "content": build_system_prompt(policy, use_plan)}, phase)
            store.append_message(run_id, {"role": "user", "content": task}, phase)
            store.add_event(run_id, "run_created", mode=mode, use_plan=use_plan)
        return run_id


    def run(self) -> str:
        while True:
            run = self._run()
            if run["status"] in PAUSED or run["status"] in TERMINAL:
                self._announce(run)
                return run["status"]
            if self._detect_uncertain():
                continue
            if self._process_pending_calls():
                continue
            run = self._run()
            if run["phase"] == "planning" and run["plan_status"] == "pending_approval":
                self.store.update_run(self.run_id, status="awaiting_plan_approval")
                continue
            self._model_step()

    def resume(self) -> str:
        run = self._run()
        if run["status"] == "model_error":
            self.store.update_run(self.run_id, status="running", status_detail=None)
        return self.run()

    def _run(self) -> dict:
        return self.store.get_run(self.run_id)

    def _say(self, text: str) -> None:
        self.out(text)


    def _detect_uncertain(self) -> bool:
        """Вызов в 'started' = процесс упал между началом и сохранением результата."""
        stuck = (self.store.calls(self.run_id, status="started")
                 + self.store.calls(self.run_id, status="uncertain"))
        if not stuck:
            return False
        with self.store.tx():
            for row in stuck:
                if row["status"] == "started":
                    self.store.update_call(self.run_id, row["call_id"], row["attempt"],
                                           status="uncertain")
                    self.store.add_event(self.run_id, "call_uncertain",
                                         call_id=row["call_id"], attempt=row["attempt"])
            self.store.update_run(self.run_id, status="awaiting_recovery",
                                  status_detail="найден вызов, прерванный сбоем")
        return True

    def recover(self, action: str, call_id: str | None = None) -> None:
        """Решение пользователя по неопределённому вызову: retry / fail / abort."""
        run = self._run()
        if run["status"] != "awaiting_recovery":
            raise UserError(f"запуск не ждёт восстановления (статус {run['status']})")
        row = self._pick(self.store.calls(self.run_id, status="uncertain"), call_id, "неопределённых")
        tool = self.registry.get(row["name"])
        with self.store.tx():
            if action == "retry":
                if tool is None or not tool.retryable:
                    raise UserError(f"инструмент {row['name']} не помечен как безопасно повторяемый: "
                                    "выбери --fail или --abort")
                self.store.update_call(self.run_id, row["call_id"], row["attempt"],
                                       status="superseded", user_decision="retry")
                self.store.insert_call(
                    run_id=self.run_id, call_id=row["call_id"], attempt=row["attempt"] + 1,
                    msg_seq=row["msg_seq"], phase=row["phase"], task_id=row["task_id"],
                    name=row["name"], args=row["args"], args_raw=row["args_raw"], kind=row["kind"],
                    status="approved", decision=row["decision"],
                    decision_reason=row["decision_reason"], user_decision="retry",
                )
            elif action == "fail":
                self.store.update_call(self.run_id, row["call_id"], row["attempt"],
                                       user_decision="fail")
                self._answer(row, "failed", {
                    "ok": False,
                    "error": "Вызов был прерван сбоем runtime; пользователь пометил шаг неуспешным. "
                             "Изменения могли примениться частично - проверь состояние файлов.",
                })
            elif action == "abort":
                self.store.update_call(self.run_id, row["call_id"], row["attempt"],
                                       status="failed", user_decision="abort")
                self.store.update_run(self.run_id, status="failed",
                                      status_detail="пользователь завершил запуск после сбоя")
            else:
                raise UserError("действие должно быть retry, fail или abort")
            self.store.add_event(self.run_id, "recovery_decision", call_id=row["call_id"],
                                 attempt=row["attempt"], action=action)
            if action != "abort" and not self.store.calls(self.run_id, status="uncertain"):
                self.store.update_run(self.run_id, status="running", status_detail=None)


    def _pending_tool_calls(self) -> tuple[int | None, list[dict]]:
        """Tool calls последнего ответа модели, на которые ещё нет сообщения role=tool."""
        msgs = self.store.messages(self.run_id)
        last = None
        for i, (_, msg) in enumerate(msgs):
            if msg["role"] == "assistant":
                last = i
        if last is None:
            return None, []
        seq, msg = msgs[last]
        answered = {m["tool_call_id"] for _, m in msgs[last + 1:] if m["role"] == "tool"}
        return seq, [tc for tc in msg.get("tool_calls") or [] if tc["id"] not in answered]

    def _process_pending_calls(self) -> bool:
        """True, если запуск остановился (пауза или конец бюджета)."""
        msg_seq, pending = self._pending_tool_calls()
        if not pending:
            return False

        for tc in pending:
            row = self.store.latest_call(self.run_id, tc["id"])
            if row is None:
                self._intake(msg_seq, tc)
            elif row["msg_seq"] != msg_seq:
                self._replay(row, tc)

        msg_seq, pending = self._pending_tool_calls()
        rows = [self.store.latest_call(self.run_id, tc["id"]) for tc in pending]

        to_run = [r for r in rows if r["status"] in EXECUTABLE | {"awaiting_approval"}]
        problem = check_batch(self._run(), [r["kind"] for r in to_run])
        if problem:
            with self.store.tx():
                for r in to_run:
                    self._answer(r, "rejected_budget", {
                        "ok": False,
                        "error": f"Бюджет: {problem}. Ни один вызов из этой пачки не выполнен.",
                        "budget_remaining": remaining(self._run()),
                    })
                self.store.add_event(self.run_id, "batch_rejected_budget", reason=problem,
                                     calls=[r["call_id"] for r in to_run])
            self._say(f"[budget] пачка из {len(to_run)} вызовов отклонена: {problem}")
            rem = remaining(self._run())
            if rem["tool_calls"] <= 0 or rem["output_bytes"] <= 0:
                self._stop_budget(problem)
                return True
            return False

        for row in rows:
            if row["status"] in EXECUTABLE:
                self._execute(row)
            elif row["status"] == "awaiting_approval":
                with self.store.tx():
                    self.store.update_run(self.run_id, status="awaiting_approval",
                                          status_detail=row["call_id"])
                    self.store.add_event(self.run_id, "approval_requested", call_id=row["call_id"])
                return True
        return False

    def _intake(self, msg_seq: int, tc: dict) -> None:
        """Новый tool call: проверка имени, JSON, схемы и policy. Ничего не выполняет."""
        run = self._run()
        name = tc["function"]["name"]
        raw = tc["function"].get("arguments") or "{}"
        tool = self.registry.get(name)
        args, error = None, None
        try:
            args = json.loads(raw)
        except json.JSONDecodeError as exc:
            error = f"аргументы не являются корректным JSON: {exc}"
        if tool is None:
            error = f"неизвестный инструмент '{name}'. Доступны: {', '.join(self.registry.names())}"
        elif error is None:
            error = self.registry.validate(tool, args)

        row = {
            "run_id": self.run_id, "call_id": tc["id"], "attempt": 1, "msg_seq": msg_seq,
            "phase": run["phase"],
            "task_id": run["current_task"] if run["phase"] == "execution" else None,
            "name": name, "args": args if isinstance(args, dict) else None, "args_raw": raw,
            "kind": tool.kind.value if tool else "unknown",
        }
        with self.store.tx():
            self.store.insert_call(**row, status="proposed")
            if error:
                self._answer(row, "denied", {"ok": False, "error": error})
                self._say(f"[runtime] {name}: ошибка вызова - {error}")
                return
            decision = self.policy.check(tool, args, mode=run["mode"], phase=run["phase"])
            self.store.update_call(self.run_id, tc["id"], 1, decision=decision.action,
                                   decision_reason=decision.reason)
            self.store.add_event(self.run_id, "policy_decision", call_id=tc["id"], tool=name,
                                 action=decision.action, reason=decision.reason)
            if decision.action == "deny":
                self._answer(row, "denied", {"ok": False, "error": f"Отклонено policy: {decision.reason}"})
                self._say(f"[policy] DENY {name}({_short(args)}): {decision.reason}")
            elif decision.action == "allow":
                self.store.update_call(self.run_id, tc["id"], 1, status="allowed")
            else:
                self.store.update_call(self.run_id, tc["id"], 1, status="awaiting_approval")
                self._say(f"[policy] ASK {name}: {decision.reason}")

    def _replay(self, row: dict, tc: dict) -> None:
        """Модель повторила tool_call_id из прошлого ответа: файлы повторно не меняем."""
        name = tc["function"]["name"]
        raw = tc["function"].get("arguments") or "{}"
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            args = None
        same = name == row["name"] and _canonical(args, raw) == _canonical(row["args"], row["args_raw"])
        if same and row["content"] is not None:
            content = row["content"]
            note = "повтор tool_call_id: возвращён сохранённый результат, действие не выполнялось"
        else:
            content = json.dumps({
                "ok": False,
                "error": f"Ошибка протокола: tool_call_id '{tc['id']}' уже использован "
                         f"с другим инструментом или аргументами. Вызов не выполнен.",
            }, ensure_ascii=False)
            note = "ошибка протокола: tool_call_id повторён с другими данными"
        with self.store.tx():
            self.store.append_message(self.run_id, {"role": "tool", "tool_call_id": tc["id"],
                                                    "content": content}, self._run()["phase"])
            self.store.add_event(self.run_id, "tool_call_replay", call_id=tc["id"], same=same)
        self._say(f"[runtime] {name}: {note}")

    def _execute(self, row: dict) -> None:
        run = self._run()
        tool = self.registry.get(row["name"])
        error = self.registry.validate(tool, row["args"])
        decision = self.policy.check(tool, row["args"], mode=run["mode"], phase=run["phase"])
        if error or decision.action == "deny":
            with self.store.tx():
                self._answer(row, "denied", {"ok": False, "error": error or f"Отклонено policy: {decision.reason}"})
            return
        if decision.action == "ask" and row["user_decision"] not in ("approved", "retry"):
            with self.store.tx():
                self.store.update_call(self.run_id, row["call_id"], row["attempt"],
                                       status="awaiting_approval")
            return

        def make_ctx(max_output: int, task_id, call_id) -> ToolContext:
            return ToolContext(workspace=self.workspace, policy=self.policy, max_output=max_output,
                               bash_timeout=self.bash_timeout, runtime=self, run_id=self.run_id,
                               task_id=task_id, call_id=call_id)

        try:
            payload = self.executor.execute(self.run_id, row, tool, make_ctx)
        except BudgetExceeded as exc:
            with self.store.tx():
                self._answer(row, "rejected_budget", {"ok": False, "error": f"Бюджет: {exc}"})
            self._say(f"[budget] {row['name']} не выполнен: {exc}")
            return
        self._say(f"[tool] {row['name']}({_short(row['args'], 80)}) -> {self._result_summary(payload)}")

    def _answer(self, row: dict, status: str, payload: dict) -> None:
        """Завершить вызов без выполнения и отдать модели результат role=tool."""
        content = json.dumps(payload, ensure_ascii=False)
        self.store.update_call(self.run_id, row["call_id"], row.get("attempt", 1), status=status,
                               result=payload, content=content, finished_at=now())
        self.store.append_message(self.run_id, {"role": "tool", "tool_call_id": row["call_id"],
                                                "content": content}, self._run()["phase"])

    @staticmethod
    def _result_summary(payload: dict) -> str:
        if not payload.get("ok"):
            return f"ERROR: {_short(payload.get('error', payload), 120)}"
        if "exit_code" in payload:
            return f"exit_code={payload['exit_code']}"
        return "ok" + (" (вывод обрезан)" if payload.get("truncated_by_runtime") else "")


    def _pick(self, rows: list[dict], call_id: str | None, what: str) -> dict:
        if call_id:
            for row in rows:
                if row["call_id"] == call_id:
                    return row
            raise UserError(f"нет {what} вызовов с id {call_id}")
        if not rows:
            raise UserError(f"нет {what} вызовов")
        return rows[0]

    def approve(self, call_id: str | None = None) -> None:
        run = self._run()
        if run["status"] != "awaiting_approval":
            raise UserError(f"запуск не ждёт подтверждения (статус {run['status']})")
        row = self._pick(self.store.calls(self.run_id, status="awaiting_approval"), call_id, "ожидающих")
        with self.store.tx():
            self.store.update_call(self.run_id, row["call_id"], row["attempt"],
                                   status="approved", user_decision="approved")
            self.store.add_event(self.run_id, "user_approved", call_id=row["call_id"])
            self.store.update_run(self.run_id, status="running", status_detail=None)

    def deny(self, call_id: str | None = None, reason: str = "без объяснения") -> None:
        run = self._run()
        if run["status"] != "awaiting_approval":
            raise UserError(f"запуск не ждёт подтверждения (статус {run['status']})")
        row = self._pick(self.store.calls(self.run_id, status="awaiting_approval"), call_id, "ожидающих")
        with self.store.tx():
            self.store.update_call(self.run_id, row["call_id"], row["attempt"],
                                   user_decision="denied", decision_reason=reason)
            self._answer(row, "denied", {"ok": False,
                                         "error": f"Пользователь отклонил действие. Причина: {reason}"})
            self.store.add_event(self.run_id, "user_denied", call_id=row["call_id"], reason=reason)
            self.store.update_run(self.run_id, status="running", status_detail=None)


    def submit_plan(self, plan: dict) -> ToolResult:
        """Вызывается инструментом submit_plan. Сохраняет план, ничего не исполняет."""
        if self._run()["phase"] != "planning":
            return ToolResult(False, {"error": "план уже утверждён, повторно сдавать не нужно"})
        problems = validate_plan(plan, self.policy, self.registry)
        if problems:
            return ToolResult(False, {"error": "План отклонён runtime, исправь и вызови submit_plan снова",
                                      "problems": problems})
        with self.store.tx():
            self.store.update_run(self.run_id, plan=plan, plan_status="pending_approval")
            self.store.add_event(self.run_id, "plan_submitted", steps=len(plan["steps"]))
        return ToolResult(True, {"status": "saved",
                                 "message": "План сохранён и ждёт решения пользователя."})

    def approve_plan(self) -> None:
        run = self._run()
        if run["status"] != "awaiting_plan_approval":
            raise UserError(f"запуск не ждёт утверждения плана (статус {run['status']})")
        with self.store.tx():
            self.store.update_run(self.run_id, plan_status="approved", phase="execution",
                                  status="running", status_detail=None)
            create_tasks(self.store, self.run_id, run["plan"])
            first = next_ready(self.store.tasks(self.run_id))
            start_task(self.store, self.run_id, first, self.workspace)
            self.store.append_message(self.run_id, {
                "role": "user",
                "content": "План утверждён пользователем. Переходим к исполнению: задачи выполняются "
                           "по одной.\n" + task_brief(self.store.get_task(self.run_id, first["task_id"])),
            }, "execution")
            self.store.add_event(self.run_id, "plan_approved")
        self._print_checklist()

    def reject_plan(self, reason: str = "без объяснения") -> None:
        run = self._run()
        if run["status"] != "awaiting_plan_approval":
            raise UserError(f"запуск не ждёт утверждения плана (статус {run['status']})")
        with self.store.tx():
            self.store.update_run(self.run_id, plan_status="rejected", status="running", status_detail=None)
            self.store.append_message(self.run_id, {
                "role": "user",
                "content": f"Пользователь отклонил план. Причина: {reason}. "
                           "Составь новый план и снова вызови submit_plan.",
            }, "planning")
            self.store.add_event(self.run_id, "plan_rejected", reason=reason, plan=run["plan"])


    def finish_task(self, task_id: str, summary: str) -> ToolResult:
        """Вызывается инструментом finish_task. Задача done только после проверки runtime."""
        task = self.store.get_task(self.run_id, task_id)
        if task is None:
            return ToolResult(False, {"error": f"задачи {task_id} нет в плане"})
        if task["status"] == "done":  # идемпотентно: повтор после сбоя
            return ToolResult(True, {"task_id": task_id, "status": "done",
                                     "evidence": task["evidence"], "note": "задача уже подтверждена"})
        run = self._run()
        if task_id != run["current_task"]:
            return ToolResult(False, {"error": f"сейчас выполняется задача {run['current_task']}, "
                                               f"а не {task_id}"})
        ok, info = verify_task(self.store, run, task, self.policy)
        if not ok:
            self.store.add_event(self.run_id, "task_verification_failed", task_id=task_id, reason=info)
            self._say(f"[task] {task_id}: runtime НЕ подтвердил результат - {info}")
            return ToolResult(False, {"error": f"Runtime не подтвердил результат задачи {task_id}: {info}"})

        with self.store.tx():
            self.store.update_task(self.run_id, task_id, status="done", evidence=info,
                                   summary=summary, finished_at=now())
            self.store.add_event(self.run_id, "task_done", task_id=task_id, evidence=info)
            nxt = next_ready(self.store.tasks(self.run_id))
            if nxt:
                start_task(self.store, self.run_id, nxt, self.workspace)
            else:
                self.store.update_run(self.run_id, current_task=None)
        self._print_checklist()
        data = {"task_id": task_id, "status": "done", "evidence": info}
        if nxt:
            data["next_task"] = task_brief(self.store.get_task(self.run_id, nxt["task_id"]))
        else:
            data["message"] = "Все задачи подтверждены. Дай короткий финальный ответ без вызова инструментов."
        return ToolResult(True, data)

    def _unfinished_tasks(self) -> list[dict]:
        return [t for t in self.store.tasks(self.run_id) if t["status"] != "done"]

    def _print_checklist(self) -> None:
        tasks = self.store.tasks(self.run_id)
        if tasks:
            self._say("Задачи:\n" + checklist(tasks))


    def _model_step(self) -> None:
        run = self._run()
        try:
            charge_model_request(self.store, self.run_id)
        except BudgetExceeded as exc:
            self._stop_budget(str(exc))
            return
        self.crash("after_model_charge")

        with_tasks = bool(run["use_plan"]) and run["phase"] == "execution"
        tools = self.registry.specs(run["phase"], with_tasks)
        messages = [m for _, m in self.store.messages(self.run_id)]
        used = self._run()["budget_used"]["model_requests"]
        self._say(f"[model] запрос {used}/{run['budget_limits']['limits']['model_requests']} "
                  f"(фаза {run['phase']})")
        try:
            response = self.llm.chat(messages, tools)
        except LLMError as exc:
            with self.store.tx():
                self.store.update_run(self.run_id, status="model_error", status_detail=str(exc))
                self.store.add_event(self.run_id, "model_error", error=str(exc))
            return

        tool_calls = self._normalize_calls(response.get("tool_calls") or [])
        message = {"role": "assistant", "content": response.get("content") or ""}
        if tool_calls:
            message["tool_calls"] = tool_calls
        with self.store.tx():
            self.store.append_message(self.run_id, message, run["phase"])
        if tool_calls:
            names = ", ".join(tc["function"]["name"] for tc in tool_calls)
            self._say(f"[model] tool calls: {names}")
            return

        text = message["content"]
        if run["phase"] == "planning":
            with self.store.tx():
                self.store.append_message(self.run_id, {"role": "user", "content": PLANNING_REMINDER}, "planning")
            self._say("[runtime] модель ответила текстом в фазе планирования - напоминаю про submit_plan")
            return
        if run["use_plan"] and self._unfinished_tasks():
            current = self.store.get_task(self.run_id, run["current_task"])
            with self.store.tx():
                self.store.append_message(self.run_id, {"role": "user",
                                                        "content": task_reminder(task_brief(current))},
                                          "execution")
            self._say("[runtime] модель ответила текстом, но задачи не подтверждены - продолжаем")
            return
        with self.store.tx():
            self.store.update_run(self.run_id, status="finished", final_answer=text)
            self.store.add_event(self.run_id, "run_finished")

    @staticmethod
    def _normalize_calls(tool_calls: list[dict]) -> list[dict]:
        """Пустые и повторяющиеся внутри ОДНОГО ответа id заменяем уникальными."""
        seen, result = set(), []
        for tc in tool_calls:
            cid = tc.get("id") or f"call_{uuid.uuid4().hex[:10]}"
            if cid in seen:
                cid = f"{cid}_{uuid.uuid4().hex[:6]}"
            seen.add(cid)
            result.append({"id": cid, "type": "function",
                           "function": {"name": tc["function"]["name"],
                                        "arguments": tc["function"].get("arguments") or "{}"}})
        return result

    def _stop_budget(self, reason: str) -> None:
        with self.store.tx():
            self.store.update_run(self.run_id, status="budget_exhausted", status_detail=reason)
            self.store.add_event(self.run_id, "budget_exhausted", reason=reason)


    def call_preview(self, row: dict) -> str:
        args = row["args"] or {}
        if row["name"] == "write_file":
            try:
                path = self.policy.check_path(args["path"])
                old = path.read_text(encoding="utf-8") if path.is_file() else ""
            except (PolicyViolation, OSError):
                old = ""
            diff = "".join(difflib.unified_diff(
                old.splitlines(True), args["content"].splitlines(True),
                fromfile=f"a/{args['path']}", tofile=f"b/{args['path']}"))
            return f"write_file {args['path']}\n{diff or '(файл уже совпадает с этим содержимым)'}"
        if row["name"] == "Bash":
            return f"Bash: `{args.get('command')}` (cwd={args.get('cwd')})"
        return f"{row['name']}({_short(args, 300)})"

    def _announce(self, run: dict) -> None:
        rid, status = self.run_id, run["status"]
        self._say(f"[run {rid}] статус: {status}; бюджет: {format_budget(run)}")
        if status == "awaiting_approval":
            row = self.store.calls(rid, status="awaiting_approval")[0]
            self._say("Требуется подтверждение действия:\n" + self.call_preview(row))
            self._say(f"Причина: {row['decision_reason']}")
            self._say(f"  разрешить:  python -m agent approve {rid}\n"
                      f"  отклонить:  python -m agent deny {rid} --reason \"...\"")
        elif status == "awaiting_plan_approval":
            self._say("План ждёт решения пользователя:\n" + render_plan(run["plan"]))
            self._say(f"  утвердить:  python -m agent plan {rid} --approve\n"
                      f"  отклонить:  python -m agent plan {rid} --reject --reason \"...\"")
        elif status == "awaiting_recovery":
            for row in self.store.calls(rid, status="uncertain"):
                tool = self.registry.get(row["name"])
                retry = "можно повторить" if tool and tool.retryable else "повтор запрещён"
                self._say(f"Неопределённый вызов {row['call_id']} (попытка {row['attempt']}): "
                          f"{self.call_preview(row)}\n  [{retry}]")
            self._say(f"  python -m agent recover {rid} --retry | --fail | --abort")
        elif status == "finished":
            self._print_checklist()
            self._say("Финальный ответ модели:\n" + (run["final_answer"] or ""))
        elif status in ("failed", "budget_exhausted", "model_error"):
            self._say(f"Причина: {run['status_detail']}")
            if status == "model_error":
                self._say(f"  продолжить: python -m agent resume {rid}")
