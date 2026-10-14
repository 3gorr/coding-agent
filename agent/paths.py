from pathlib import Path


class PathError(ValueError):
    """Путь некорректен или ведёт за пределы workspace."""


def is_within(path: Path, root: Path) -> bool:
    """True, если path совпадает с root или лежит внутри него."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_inside(root: Path, user_path: str) -> Path:
    """Возвращает канонический путь внутри root или бросает PathError."""
    if not isinstance(user_path, str) or not user_path.strip():
        raise PathError("путь должен быть непустой строкой")
    if "\x00" in user_path:
        raise PathError("путь содержит нулевой байт")
    raw = Path(user_path)
    if raw.is_absolute():
        raise PathError(
            f"абсолютный путь '{user_path}' запрещён: укажите путь относительно workspace"
        )
    root = root.resolve()
    candidate = (root / raw).resolve(strict=False)
    if not is_within(candidate, root):
        raise PathError(
            f"путь '{user_path}' ведёт за пределы workspace (через '..' или симлинк)"
        )
    return candidate


def rel(root: Path, path: Path) -> str:
    """Путь относительно root в POSIX-виде ('.' для самого root)."""
    return path.relative_to(root.resolve()).as_posix() or "."
