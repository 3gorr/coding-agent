"""Файловые инструменты.

Каждый обработчик ЗАНОВО проверяет путь через ctx.policy.check_path, даже если
runtime уже проверил его до выполнения: инструмент не доверяет вызывающему.
"""
import os
from pathlib import Path

from ..snapshot import SKIP_DIRS
from .base import Tool, ToolContext, ToolError, ToolKind, ToolResult

MAX_READ_BYTES = 200_000
MAX_LIST_ENTRIES = 500


def _list_files(args: dict, ctx: ToolContext) -> ToolResult:
    base = ctx.policy.check_path(args["path"])
    if not base.exists():
        raise ToolError(f"'{args['path']}' не существует")
    root = ctx.workspace
    if base.is_file():
        return ToolResult(True, {"path": args["path"], "entries": [base.relative_to(root).as_posix()]})

    entries, hidden = [], 0
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        current = Path(dirpath)
        kept = []
        for name in sorted(dirnames):
            path = current / name
            if name in SKIP_DIRS:
                continue
            if path.is_symlink() or not ctx.policy.is_visible(path):
                hidden += 1  # симлинки на директории и запрещённые директории не раскрываем
                continue
            kept.append(name)
            entries.append(path.relative_to(root).as_posix() + "/")
        dirnames[:] = kept
        for name in sorted(filenames):
            path = current / name
            if path.suffix == ".pyc":
                continue
            if not ctx.policy.is_visible(path):
                hidden += 1
                continue
            entries.append(path.relative_to(root).as_posix())

    entries.sort()
    return ToolResult(True, {
        "path": args["path"],
        "entries": entries[:MAX_LIST_ENTRIES],
        "truncated": len(entries) > MAX_LIST_ENTRIES,
        "hidden_by_policy": hidden,
    })


def _read_file(args: dict, ctx: ToolContext) -> ToolResult:
    path = ctx.policy.check_path(args["path"])
    if not path.exists():
        raise ToolError(f"файл '{args['path']}' не найден")
    if not path.is_file():
        raise ToolError(f"'{args['path']}' - не файл")
    with path.open("rb") as fh:
        data = fh.read(MAX_READ_BYTES + 1)
    return ToolResult(True, {
        "path": args["path"],
        "content": data[:MAX_READ_BYTES].decode("utf-8", errors="replace"),
        "truncated": len(data) > MAX_READ_BYTES,
    })


def _write_file(args: dict, ctx: ToolContext) -> ToolResult:
    path = ctx.policy.check_path(args["path"])
    if path.exists() and not path.is_file():
        raise ToolError(f"'{args['path']}' существует и не является файлом")
    created = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"], encoding="utf-8")
    return ToolResult(True, {
        "path": args["path"],
        "created": created,
        "bytes_written": len(args["content"].encode("utf-8")),
    })


def _delete_file(args: dict, ctx: ToolContext) -> ToolResult:
    path = ctx.policy.check_path(args["path"])
    if path.is_symlink() or not path.is_file():
        raise ToolError(f"можно удалить только существующий обычный файл, '{args['path']}' не подходит")
    path.unlink()
    return ToolResult(True, {"path": args["path"], "deleted": True})


def _path_schema(description: str) -> dict:
    return {"type": "string", "minLength": 1, "description": description}


LIST_FILES = Tool(
    name="list_files",
    description="Рекурсивно показать файлы внутри директории workspace. "
                "Путь относительный, '.' - корень workspace.",
    parameters={
        "type": "object",
        "properties": {"path": _path_schema("Директория относительно корня workspace, например '.'")},
        "required": ["path"],
        "additionalProperties": False,
    },
    kind=ToolKind.READ,
    retryable=True,
    handler=_list_files,
    path_args=("path",),
)

READ_FILE = Tool(
    name="read_file",
    description="Прочитать текстовый файл из workspace.",
    parameters={
        "type": "object",
        "properties": {"path": _path_schema("Путь к файлу относительно корня workspace")},
        "required": ["path"],
        "additionalProperties": False,
    },
    kind=ToolKind.READ,
    retryable=True,
    handler=_read_file,
    path_args=("path",),
)

# Запись ВСЕГО содержимого файла идемпотентна: повтор с теми же аргументами
# приводит к тому же результату, поэтому инструмент можно безопасно повторить.
WRITE_FILE = Tool(
    name="write_file",
    description="Создать файл или полностью перезаписать его новым содержимым.",
    parameters={
        "type": "object",
        "properties": {
            "path": _path_schema("Путь к файлу относительно корня workspace"),
            "content": {"type": "string", "description": "Полное новое содержимое файла"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    kind=ToolKind.WRITE,
    retryable=True,
    handler=_write_file,
    path_args=("path",),
    phases=("execution",),
)

# Удаление не повторяем автоматически: после сбоя неизвестно, что лежит на
# месте файла, решение всегда за пользователем (--fail или --abort).
DELETE_FILE = Tool(
    name="delete_file",
    description="Удалить файл в workspace (всегда требует подтверждения пользователя).",
    parameters={
        "type": "object",
        "properties": {"path": _path_schema("Путь к файлу относительно корня workspace")},
        "required": ["path"],
        "additionalProperties": False,
    },
    kind=ToolKind.DELETE,
    retryable=False,
    handler=_delete_file,
    path_args=("path",),
    phases=("execution",),
)

FS_TOOLS = [LIST_FILES, READ_FILE, WRITE_FILE, DELETE_FILE]
