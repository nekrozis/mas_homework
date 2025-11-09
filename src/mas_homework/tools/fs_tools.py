# src/mas_homework/tools/fs_tools.py
import os
import glob as glob_module

# 假设 ToolResult 在 types.py 中定义为 TypedDict，因此可作为构造函数使用
from ..tool_system import GLOBAL_TOOLS, tool
from ..types import ToolResult 

# ==============================================================================
# 文件系统工具定义
# ==============================================================================

@tool(GLOBAL_TOOLS, "ls", "List files and directories using path_or_pattern parameter. E.g., path_or_pattern='*.pdf'")
def ls(path_or_pattern: str = ".") -> ToolResult:
    """列出文件和目录"""
    output: str = "" # 统一初始化 output 变量
    
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
            # 路径不存在时立即返回
            return ToolResult(
                success=False,
                output=None,
                error=f"Path '{path_or_pattern}' not found",
            )

        # 成功路径：返回 output 变量的值
        return ToolResult(success=True, output=output, error=None)
        
    except Exception as e:
        # 异常路径
        return ToolResult(
            success=False,
            output=None,
            error=f"Error listing files: {type(e).__name__}: {e}",
        )

@tool(GLOBAL_TOOLS, "finish", "Signal task completion. Parameter 'answer' must contain the final structured JSON result.")
def finish(answer: str) -> ToolResult:
    """完成任务的特殊工具，标记最终答案"""
    # 使用 ToolResult 构造函数
    return ToolResult(success=True, output=f"TASK_COMPLETED:{answer}", error=None)