import hashlib
import os
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".git"}


def file_sha(path: Path) -> str | None:
    """sha256 содержимого файла или None, если файла нет."""
    if path.is_symlink() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_tree(root: Path) -> dict[str, str]:
    """{относительный путь: sha256} для всех обычных файлов внутри root.

    Симлинки не читаются (они могут вести наружу), *.pyc и служебные
    директории пропускаются.
    """
    root = Path(root).resolve()
    result: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix == ".pyc" or path.is_symlink():
                continue
            result[path.relative_to(root).as_posix()] = file_sha(path)
    return result
