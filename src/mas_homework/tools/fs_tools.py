# src/mas_homework/tools/fs_tools.py
import glob as glob_module
from pathlib import Path
from typing import List

from ..tool_system import GLOBAL_TOOLS, tool
from ..types import ToolResult


def _format_item(path: Path) -> str:
    """根据路径类型标记为文件或目录"""
    if path.is_file():
        return f"[FILE] {path.as_posix()}"
    elif path.is_dir():
        return f"[DIR] {path.as_posix()}/"
    return path.as_posix()


@tool(
    GLOBAL_TOOLS,
    "ls",
    "List files and directories using path_or_pattern. E.g., 'data/*.pdf' or 'data/'.",
)
def ls(path_or_pattern: str = ".") -> ToolResult:
    """列出文件和目录，支持通配符或目录路径"""
    try:
        # 支持通配符
        if any(c in path_or_pattern for c in "*?[]"):
            matches: List[Path] = [Path(p) for p in glob_module.glob(path_or_pattern)]
        # 支持目录
        elif Path(path_or_pattern).is_dir():
            matches = list(Path(path_or_pattern).iterdir())
        # 单文件
        elif Path(path_or_pattern).is_file():
            matches = [Path(path_or_pattern)]
        else:
            return ToolResult(
                success=False,
                output=None,
                error=f"Path '{path_or_pattern}' not found or is invalid.",
            )

        if not matches:
            output_msg = (
                f"Directory '{path_or_pattern}' is empty."
                if Path(path_or_pattern).is_dir()
                else f"No files found matching pattern/path: {path_or_pattern}"
            )
            return ToolResult(success=True, output=output_msg)

        output = "\n".join(_format_item(m) for m in sorted(matches))
        return ToolResult(success=True, output=output)

    except Exception as e:
        return ToolResult(
            success=False,
            output=None,
            error=f"Error listing files: {type(e).__name__}: {e}",
        )
