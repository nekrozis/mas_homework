# src/mas_homework/tools/fs_tools.py
import glob as glob_module
import os

from ..tool_system import GLOBAL_TOOLS, ToolResult, tool

# ==============================================================================
# 文件系统工具定义
# ==============================================================================


@tool(
    GLOBAL_TOOLS,
    "ls",
    "List files and directories using path_or_pattern parameter. E.g., path_or_pattern='*.pdf'",
)
def ls(path_or_pattern: str = ".") -> ToolResult:
    """列出文件和目录"""
    try:
        # 优先处理通配符
        if "*" in path_or_pattern or "?" in path_or_pattern or "[" in path_or_pattern:
            matches = glob_module.glob(path_or_pattern)
            result = [
                f"[FILE] {match}" if os.path.isfile(match) else f"[DIR] {match}/"
                for match in sorted(matches)
            ]
            output = (
                "\n".join(result)
                if result
                else f"No files found matching pattern: {path_or_pattern}"
            )
        # 处理目录
        elif os.path.isdir(path_or_pattern):
            items = sorted(os.listdir(path_or_pattern))
            result = [
                f"[FILE] {item}"
                if os.path.isfile(os.path.join(path_or_pattern, item))
                else f"[DIR] {item}/"
                for item in items
            ]
            output = (
                "\n".join(result)
                if items
                else f"Directory '{path_or_pattern}' is empty"
            )
        # 处理单个文件
        elif os.path.isfile(path_or_pattern):
            output = f"[FILE] {path_or_pattern}"
        else:
            return {
                "success": False,
                "output": None,
                "error": f"Path '{path_or_pattern}' not found",
            }

        return {"success": True, "output": output, "error": None}
    except Exception as e:
        return {
            "success": False,
            "output": None,
            "error": f"Error listing files: {type(e).__name__}: {e}",
        }


@tool(
    GLOBAL_TOOLS,
    "finish",
    "Signal task completion. Parameter 'answer' must contain the final structured JSON result (the output of pdf_analysis).",
)
def finish(answer: str) -> ToolResult:
    """完成任务的特殊工具，标记最终答案"""
    # 使用特殊前缀确保 Agent 逻辑能识别最终答案
    return {"success": True, "output": f"TASK_COMPLETED:{answer}", "error": None}


# 注意：这个模块自身不需要执行，它的作用是注册工具。
