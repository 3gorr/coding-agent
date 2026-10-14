from .bash import BASH
from .control import CONTROL_TOOLS
from .fs import FS_TOOLS
from .registry import ToolRegistry


def default_registry() -> ToolRegistry:
    return ToolRegistry([*FS_TOOLS, BASH, *CONTROL_TOOLS])
