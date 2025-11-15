# src/mas_homework/tools/__init__.py
from ..tool_system import GLOBAL_TOOLS
from . import fs_tools, pdf_tools  # 导入工具模块，自动注册到 GLOBAL_TOOLS

__all__ = ["GLOBAL_TOOLS", "fs_tools", "pdf_tools"]
