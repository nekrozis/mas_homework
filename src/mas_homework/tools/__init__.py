# src/mas_homework/tools/__init__.py
from ..tool_system import GLOBAL_TOOLS

# 导入所有工具模块。
# 导入操作会自动执行模块内的 @tool 装饰器，将工具注册到 GLOBAL_TOOLS。
from . import fs_tools, pdf_tools

# 导出 GLOBAL_TOOLS，方便其他模块（如 agent_core.py）使用
__all__ = ["GLOBAL_TOOLS", "fs_tools", "pdf_tools"]
