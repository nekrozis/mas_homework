# src/mas_homework/tools/fs_tools.py
import glob as glob_module
import os
from pathlib import Path
from typing import Optional, Union, List

from ..tool_system import GLOBAL_TOOLS, tool
from ..types import ToolResult

# ==============================================================================
# 文件系统工具
# ==============================================================================


def _format_item(path: Path) -> str:
    """根据路径类型标记为文件或目录"""
    if path.is_file():
        return f"[FILE] {path.as_posix()}"
    elif path.is_dir():
        # 确保目录路径以斜杠结尾
        return f"[DIR] {path.as_posix()}/"
    return path.as_posix()


@tool(
    GLOBAL_TOOLS,
    "ls",
    "List files and directories using path_or_pattern. E.g., 'data/*.pdf' or 'data/'. Returns a list of paths.",
)
def ls(path_or_pattern: str = ".") -> ToolResult:
    """列出文件和目录。接受通配符或目录路径。"""
    try:
        # 使用 glob 统一处理通配符和目录
        
        # 1. 检查是否包含通配符
        if any(c in path_or_pattern for c in "*?[]"):
            matches: List[Path] = [Path(p) for p in glob_module.glob(path_or_pattern)]
        
        # 2. 检查是否是目录
        elif Path(path_or_pattern).is_dir():
            # 列出目录下的所有项
            dir_path = Path(path_or_pattern)
            matches = list(dir_path.iterdir())
        
        # 3. 检查是否是单个文件
        elif Path(path_or_pattern).is_file():
            matches = [Path(path_or_pattern)]
        
        # 4. 路径不存在
        else:
            return ToolResult(
                success=False, output=None, error=f"Path '{path_or_pattern}' not found or is invalid."
            )

        if not matches:
            # 区分空目录和无匹配项
            if Path(path_or_pattern).is_dir():
                 output_msg = f"Directory '{path_or_pattern}' is empty."
            else:
                 output_msg = f"No files found matching pattern/path: {path_or_pattern}"
                 
            return ToolResult(
                success=True,  # 即使是空目录，操作本身也是成功的
                output=output_msg,
                error=None,
            )

        # 格式化输出
        output = "\n".join(_format_item(m) for m in sorted(matches))
        return ToolResult(success=True, output=output, error=None)

    except Exception as e:
        # 捕获所有运行时错误
        return ToolResult(
            success=False,
            output=None,
            error=f"Error listing files: {type(e).__name__}: {e}",
        )


# 移除 finish 工具，因为它已被 agent_core.py 中的 final_answer 机制取代。
# -----------------------------------------------------------------------------
# @tool(
#     GLOBAL_TOOLS,
#     "finish",
#     "Signal task completion. Parameter 'answer' must contain the final structured JSON result.",
# )
# def finish(answer: str) -> ToolResult:
#     """完成任务的特殊工具，标记最终答案"""
#     return ToolResult(success=True, output=f"TASK_COMPLETED:{answer}", error=None)
# -----------------------------------------------------------------------------