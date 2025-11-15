import glob as glob_module
import os

from ..tool_system import GLOBAL_TOOLS, tool
from ..types import ToolResult

# ==============================================================================
# 文件系统工具
# ==============================================================================


def _format_item(path: str) -> str:
    """根据路径类型标记为文件或目录"""
    if os.path.isfile(path):
        return f"[FILE] {path}"
    elif os.path.isdir(path):
        return f"[DIR] {path}/"
    return path


@tool(
    GLOBAL_TOOLS,
    "ls",
    "List files and directories using path_or_pattern. E.g., '*.pdf'",
)
def ls(path_or_pattern: str = ".") -> ToolResult:
    """列出文件和目录"""
    try:
        # 1️⃣ 通配符模式
        if any(c in path_or_pattern for c in "*?[]"):
            matches = glob_module.glob(path_or_pattern)
            if not matches:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"No files found matching pattern: {path_or_pattern}",
                )
            output = "\n".join(_format_item(m) for m in sorted(matches))
            return ToolResult(success=True, output=output, error=None)

        # 2️⃣ 目录路径
        elif os.path.isdir(path_or_pattern):
            items = sorted(os.listdir(path_or_pattern))
            if not items:
                return ToolResult(
                    success=True,
                    output=f"Directory '{path_or_pattern}' is empty",
                    error=None,
                )
            output = "\n".join(
                _format_item(os.path.join(path_or_pattern, item)) for item in items
            )
            return ToolResult(success=True, output=output, error=None)

        # 3️⃣ 单个文件
        elif os.path.isfile(path_or_pattern):
            return ToolResult(
                success=True, output=f"[FILE] {path_or_pattern}", error=None
            )

        # 4️⃣ 不存在路径
        else:
            return ToolResult(
                success=False, output=None, error=f"Path '{path_or_pattern}' not found"
            )

    except Exception as e:
        return ToolResult(
            success=False,
            output=None,
            error=f"Error listing files: {type(e).__name__}: {e}",
        )


@tool(
    GLOBAL_TOOLS,
    "finish",
    "Signal task completion. Parameter 'answer' must contain the final structured JSON result.",
)
def finish(answer: str) -> ToolResult:
    """完成任务的特殊工具，标记最终答案"""
    return ToolResult(success=True, output=f"TASK_COMPLETED:{answer}", error=None)
