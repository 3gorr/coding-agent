from .paths import rel
from .plan import describe_verification
from .policy import PolicyViolation
from .snapshot import file_sha, hash_tree
from .store import now
from .tools.base import ToolKind


def create_tasks(store, run_id: str, plan: dict) -> None:
    for position, step in enumerate(plan["steps"]):
        store.insert_task(
            run_id=run_id, task_id=step["id"], position=position,
            description=step["description"], tools=step["tools"],
            depends_on=step["depends_on"], verification=step["verification"],
            status="pending",
        )


def next_ready(tasks: list[dict]) -> dict | None:
    """Первая по порядку задача, у которой все зависимости выполнены."""
    done = {t["task_id"] for t in tasks if t["status"] == "done"}
    for task in tasks:
        if task["status"] == "pending" and all(d in done for d in task["depends_on"]):
            return task
    return None


def start_task(store, run_id: str, task: dict, workspace) -> None:
    store.update_task(run_id, task["task_id"], status="in_progress",
                      start_hashes=hash_tree(workspace), started_at=now())
    store.update_run(run_id, current_task=task["task_id"])


def task_brief(task: dict) -> str:
    return (
        f"Текущая задача {task['task_id']}: {task['description']}\n"
        f"Инструменты по плану: {', '.join(task['tools']) or '-'}\n"
        f"Проверка runtime: {describe_verification(task['verification'])}\n"
        f"Когда закончишь, вызови finish_task с task_id=\"{task['task_id']}\"."
    )


def checklist(tasks: list[dict]) -> str:
    lines = []
    for task in tasks:
        mark = "x" if task["status"] == "done" else " "
        deps = f" (после: {', '.join(task['depends_on'])})" if task["depends_on"] else ""
        state = "  <- в работе" if task["status"] == "in_progress" else ""
        lines.append(f"- [{mark}] {task['task_id']}: {task['description']}{deps}{state}")
    return "\n".join(lines)


def _ok(row: dict) -> bool:
    return bool(row.get("result") and row["result"].get("ok"))


def _resolve(policy, path: str):
    try:
        return policy.check_path(path)
    except PolicyViolation:
        return None


def verify_task(store, run: dict, task: dict, policy) -> tuple[bool, dict | str]:
    """Проверить результат задачи по фактам из БД и файловой системы."""
    v = task["verification"]
    vtype = v["type"]
    run_id = run["run_id"]
    ws = policy.workspace
    rows = store.calls(run_id, task_id=task["task_id"], status="done")

    if vtype == "files_read":
        got = {}
        for row in rows:
            if row["name"] == "read_file" and _ok(row):
                path = _resolve(policy, row["args"]["path"])
                if path is not None:
                    got[path] = row["call_id"]
        missing, evidence = [], {}
        for wanted in v["paths"]:
            path = _resolve(policy, wanted)
            if path is None or path not in got:
                missing.append(wanted)
            else:
                evidence[wanted] = got[path]
        if missing:
            return False, f"в рамках задачи не было успешного read_file для: {', '.join(missing)}"
        return True, {"type": vtype, "read_calls": evidence}

    if vtype in ("file_changed", "file_unchanged"):
        path = _resolve(policy, v["path"])
        if path is None:
            return False, f"путь {v['path']} запрещён policy"
        key = rel(ws, path)
        after = file_sha(path)
        if vtype == "file_changed":
            before = (task.get("start_hashes") or {}).get(key)
            if before == after:
                return False, f"файл {v['path']} не изменился с начала задачи"
        else:
            before = (run.get("start_hashes") or {}).get(key)
            if before != after:
                return False, f"файл {v['path']} отличается от состояния на старте запуска"
        return True, {"type": vtype, "path": v["path"], "sha_before": before, "sha_after": after}

    if vtype == "command_succeeds":
        cwd = _resolve(policy, v["cwd"])
        candidates = [
            row for row in rows
            if row["name"] == "Bash"
            and row["args"].get("command") == v["command"]
            and _resolve(policy, row["args"].get("cwd", "")) == cwd
            and row["result"].get("exit_code") == 0
            and not row["result"].get("timed_out")
        ]
        if not candidates:
            return False, (f"в рамках задачи нет успешного (exit_code 0) запуска "
                           f"`{v['command']}` в cwd={v['cwd']}")
        last = max(candidates, key=lambda r: r["done_seq"])
        changed_after = [
            row for row in store.calls(run_id, status="done")
            if row["kind"] in (ToolKind.WRITE.value, ToolKind.DELETE.value)
            and _ok(row) and row["done_seq"] > last["done_seq"]
        ]
        if changed_after:
            return False, "после успешного запуска команды файлы менялись - запусти её ещё раз"
        return True, {"type": vtype, "command": v["command"], "call_id": last["call_id"],
                      "attempt": last["attempt"], "exit_code": 0}

    return False, f"неизвестный тип проверки {vtype}"
