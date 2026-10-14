"""Инструмент Bash(command, cwd).

Ограничения:
  * команда повторно проверяется policy внутри обработчика;
  * cwd канонизируется и должен быть разрешённой директорией;
  * таймаут: по истечении убивается вся группа процессов;
  * размер stdout/stderr ограничен остатком бюджета вывода;
  * окружение очищено: API-ключи и прочие переменные родителя не передаются.
"""
import os
import signal
import subprocess

from .base import Tool, ToolContext, ToolError, ToolKind, ToolResult


def _cut(data: bytes, limit: int) -> tuple[str, bool]:
    if len(data) <= limit:
        return data.decode("utf-8", errors="replace"), False
    return data[:limit].decode("utf-8", errors="ignore"), True


def _child_env() -> dict:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        # Не создаём __pycache__ в workspace: тесты не должны менять дерево проекта.
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _bash(args: dict, ctx: ToolContext) -> ToolResult:
    command = args["command"]
    ctx.policy.check_command(command)
    cwd = ctx.policy.check_path(args["cwd"])
    if not cwd.is_dir():
        raise ToolError(f"рабочая директория '{args['cwd']}' не существует")

    proc = subprocess.Popen(
        ["bash", "-c", command],
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_child_env(),
        start_new_session=True,  # своя группа процессов, чтобы убить и потомков
    )
    timed_out = False
    try:
        out, err = proc.communicate(timeout=ctx.bash_timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()

    limit = max(0, ctx.max_output)
    stdout, stdout_cut = _cut(out, limit)
    stderr, stderr_cut = _cut(err, limit)
    return ToolResult(not timed_out, {
        "command": command,
        "cwd": args["cwd"],
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_cut,
        "stderr_truncated": stderr_cut,
    })


# Bash считаем безопасно повторяемым, потому что policy разрешает только точные
# команды, которые не меняют исходники и тесты (запуск тестов, ls, pwd).
# Если добавить в commands.allow команду, меняющую файлы, это решение нужно пересмотреть.
BASH = Tool(
    name="Bash",
    description="Запустить разрешённую команду через bash. Разрешены только точные "
                "простые команды из policy. Возвращает stdout, stderr и exit_code.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "minLength": 1, "description": "Точная команда из списка разрешённых"},
            "cwd": {"type": "string", "minLength": 1, "description": "Рабочая директория относительно корня workspace, обычно '.'"},
        },
        "required": ["command", "cwd"],
        "additionalProperties": False,
    },
    kind=ToolKind.BASH,
    retryable=True,
    handler=_bash,
    path_args=("cwd",),
    phases=("execution",),
)
