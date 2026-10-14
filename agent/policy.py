import json
from dataclasses import dataclass
from pathlib import Path

from .paths import PathError, is_within, rel, resolve_inside
from .tools.base import ToolKind


FORBIDDEN_SHELL_CHARS = set("|&;<>()$`\\\"'*?[]{}~!#\n\r\t")


class PolicyConfigError(Exception):
    """Ошибка в файле policy."""


class PolicyViolation(Exception):
    """Действие нарушает policy."""


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str


class Policy:
    def __init__(self, policy_path, workspace, protected=()):
        self.path = Path(policy_path).resolve()
        self.workspace = Path(workspace).resolve()
        if not self.workspace.is_dir():
            raise PolicyConfigError(f"workspace '{workspace}' не существует")
        if is_within(self.path, self.workspace):
            raise PolicyConfigError(
                "policy-файл должен лежать вне workspace, иначе агент мог бы его изменить"
            )
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PolicyConfigError(f"не удалось прочитать policy '{policy_path}': {exc}")

        self.cmd_allow = self._str_list(data, "commands", "allow")
        self.cmd_deny = self._str_list(data, "commands", "deny")
        self.dir_allow = [self._canon_dir(d) for d in self._str_list(data, "directories", "allow")]
        self.dir_deny = [self._canon_dir(d) for d in self._str_list(data, "directories", "deny")]
        self.protected = [self.path] + [Path(p).resolve() for p in protected]


    @staticmethod
    def _str_list(data, section, key) -> list[str]:
        value = data.get(section, {}).get(key, [])
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise PolicyConfigError(f"{section}.{key} должен быть списком строк")
        return value

    def _canon_dir(self, entry: str) -> Path:
        """Директории в конфиге задаются относительно workspace и канонизируются."""
        try:
            return resolve_inside(self.workspace, entry)
        except PathError as exc:
            raise PolicyConfigError(f"директория '{entry}' в policy вне workspace: {exc}")


    def check_path(self, user_path) -> Path:
        """Канонический путь, если он разрешён, иначе PolicyViolation."""
        try:
            path = resolve_inside(self.workspace, user_path)
        except PathError as exc:
            raise PolicyViolation(str(exc))
        for protected in self.protected:
            if path == protected or is_within(path, protected):
                raise PolicyViolation(f"'{user_path}' - служебный файл агента, доступ запрещён")
        for denied in self.dir_deny:
            if is_within(path, denied):
                raise PolicyViolation(
                    f"путь '{user_path}' находится в запрещённой директории "
                    f"'{rel(self.workspace, denied)}' (directories.deny)"
                )
        if not any(is_within(path, allowed) for allowed in self.dir_allow):
            raise PolicyViolation(f"путь '{user_path}' не входит в directories.allow")
        return path

    def is_visible(self, path: Path) -> bool:
        """Можно ли показывать уже найденный на диске путь (для list_files)."""
        try:
            self.check_path(rel(self.workspace, Path(path).resolve()))
            return True
        except (PolicyViolation, ValueError):
            return False

    def check_command(self, command) -> None:
        """Проверяет команду Bash ЦЕЛИКОМ, а не только первое слово."""
        if not isinstance(command, str) or not command.strip():
            raise PolicyViolation("пустая команда")
        bad = sorted(set(command) & FORBIDDEN_SHELL_CHARS)
        if bad:
            raise PolicyViolation(
                f"команда содержит запрещённый синтаксис Bash {bad}: разрешены только "
                "точные простые команды из commands.allow (без конвейеров, перенаправлений, "
                "подстановок и кавычек)"
            )
        if command in self.cmd_deny:
            raise PolicyViolation(f"команда '{command}' в commands.deny (запрет важнее разрешения)")
        if command not in self.cmd_allow:
            raise PolicyViolation(
                f"команды '{command}' нет в commands.allow; разрешены только точные строки: "
                + ", ".join(repr(c) for c in self.cmd_allow)
            )

    def check(self, tool, args: dict, *, mode: str, phase: str) -> Decision:
        if mode not in ("safe", "auto"):
            return Decision("deny", f"неизвестный режим '{mode}'")
        if phase == "planning" and tool.kind not in (ToolKind.READ, ToolKind.CONTROL):
            return Decision(
                "deny",
                "Plan Mode: до утверждения плана разрешены только инструменты чтения "
                "(менять файлы, удалять их и вызывать Bash нельзя)",
            )
        if phase not in tool.phases:
            return Decision("deny", f"инструмент {tool.name} недоступен в фазе '{phase}'")
        try:
            for name in tool.path_args:
                if name in args:
                    self.check_path(args[name])
            if tool.kind is ToolKind.BASH:
                self.check_command(args.get("command"))
        except PolicyViolation as exc:
            return Decision("deny", str(exc))

        if tool.kind in (ToolKind.READ, ToolKind.CONTROL):
            return Decision("allow", "чтение разрешено сразу")
        if tool.kind is ToolKind.DELETE:
            return Decision("ask", "удаление всегда требует подтверждения пользователя")
        if mode == "safe":
            return Decision("ask", f"режим safe: {tool.kind.value} требует подтверждения")
        return Decision("allow", "режим auto: действие прошло ограничения policy")
