from .policy import PolicyViolation

VERIFICATION_FIELDS = {
    "files_read": ("paths",),
    "file_changed": ("path",),
    "file_unchanged": ("path",),
    "command_succeeds": ("command", "cwd"),
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string", "minLength": 1, "description": "Цель работы"},
        "steps": {
            "type": "array",
            "minItems": 1,
            "description": "Шаги в предполагаемом порядке",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,32}$"},
                    "description": {"type": "string", "minLength": 1},
                    "tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Инструменты, которые понадобятся на шаге",
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "id шагов, которые должны быть завершены раньше",
                    },
                    "verification": {
                        "type": "object",
                        "description": "Как runtime проверит результат шага",
                        "properties": {
                            "type": {"type": "string", "enum": sorted(VERIFICATION_FIELDS)},
                            "paths": {"type": "array", "items": {"type": "string"}},
                            "path": {"type": "string"},
                            "command": {"type": "string"},
                            "cwd": {"type": "string"},
                        },
                        "required": ["type"],
                    },
                },
                "required": ["id", "description", "tools", "depends_on", "verification"],
            },
        },
        "risks": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "done_criteria": {"type": "string", "minLength": 1, "description": "Критерий готовности"},
    },
    "required": ["goal", "steps", "risks", "assumptions", "done_criteria"],
}


def validate_plan(plan: dict, policy, registry) -> list[str]:
    """Семантическая проверка плана (структуру уже проверила JSON Schema)."""
    errors: list[str] = []
    steps = plan["steps"]
    ids = [s["id"] for s in steps]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        errors.append(f"повторяющиеся id шагов: {duplicates}")
    known_tools = set(registry.names()) - {"submit_plan"}

    for step in steps:
        sid = step["id"]
        for tool in step["tools"]:
            if tool not in known_tools:
                errors.append(f"{sid}: неизвестный инструмент '{tool}'")
        for dep in step["depends_on"]:
            if dep == sid:
                errors.append(f"{sid}: шаг не может зависеть от самого себя")
            elif dep not in ids:
                errors.append(f"{sid}: зависимость '{dep}' не найдена среди шагов")
        errors.extend(_check_verification(sid, step["verification"], policy))

    if not errors:
        cycle = _find_cycle(steps)
        if cycle:
            errors.append(f"циклическая зависимость между шагами: {cycle}")
    return errors


def _check_verification(sid: str, v: dict, policy) -> list[str]:
    vtype = v.get("type")
    if vtype not in VERIFICATION_FIELDS:
        return [f"{sid}: неизвестный тип проверки '{vtype}'"]
    errors = [f"{sid}: в проверке {vtype} нет поля '{f}'"
              for f in VERIFICATION_FIELDS[vtype] if f not in v]
    if errors:
        return errors
    try:
        if vtype == "files_read":
            if not v["paths"]:
                return [f"{sid}: files_read требует непустой список paths"]
            for p in v["paths"]:
                policy.check_path(p)
        elif vtype in ("file_changed", "file_unchanged"):
            policy.check_path(v["path"])
        elif vtype == "command_succeeds":
            policy.check_command(v["command"])
            policy.check_path(v["cwd"])
    except PolicyViolation as exc:
        return [f"{sid}: проверка нарушает policy: {exc}"]
    return []


def _find_cycle(steps: list[dict]) -> list[str] | None:
    """Алгоритм Кана: если не все вершины удалось упорядочить, есть цикл."""
    indegree = {s["id"]: len(s["depends_on"]) for s in steps}
    children: dict[str, list[str]] = {s["id"]: [] for s in steps}
    for s in steps:
        for dep in s["depends_on"]:
            children[dep].append(s["id"])
    queue = [sid for sid, d in indegree.items() if d == 0]
    seen = 0
    while queue:
        node = queue.pop()
        seen += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if seen == len(steps):
        return None
    return sorted(sid for sid, d in indegree.items() if d > 0)


def describe_verification(v: dict) -> str:
    t = v.get("type")
    if t == "files_read":
        return f"успешно прочитаны файлы {', '.join(v['paths'])}"
    if t == "file_changed":
        return f"файл {v['path']} изменён в ходе задачи"
    if t == "file_unchanged":
        return f"файл {v['path']} не отличается от состояния на старте запуска"
    if t == "command_succeeds":
        return f"команда `{v['command']}` (cwd={v['cwd']}) завершилась с кодом 0, после неё файлы не менялись"
    return str(v)


def render_plan(plan: dict) -> str:
    lines = [f"Цель: {plan['goal']}", "Шаги:"]
    for i, s in enumerate(plan["steps"], 1):
        deps = f" (после: {', '.join(s['depends_on'])})" if s["depends_on"] else ""
        lines.append(f"  {i}. [{s['id']}] {s['description']}{deps}")
        lines.append(f"     инструменты: {', '.join(s['tools']) or '-'}")
        lines.append(f"     проверка: {describe_verification(s['verification'])}")
    lines.append("Риски: " + ("; ".join(plan["risks"]) or "-"))
    lines.append("Допущения: " + ("; ".join(plan["assumptions"]) or "-"))
    lines.append(f"Критерий готовности: {plan['done_criteria']}")
    return "\n".join(lines)
