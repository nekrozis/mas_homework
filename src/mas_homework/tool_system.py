# src/mas_homework/tool_system.py
from functools import wraps
from typing import Any, Callable, Dict, Optional

from .types import ToolResult  # 从本地包导入

# ToolFunction 是一个类型别名
ToolFunction = Callable[..., ToolResult]


class Tool:
    """工具元数据和功能封装"""

    def __init__(self, name: str, description: str, function: ToolFunction):
        self.name = name
        self.description = description
        self.function = function


class ToolSystem:
    """
    工具系统的核心：作为一个不可变的注册表管理所有工具。
    工具的执行是纯函数式的。
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """将工具注册到系统（在初始化阶段使用）"""
        self._tools[tool.name] = tool

    def get_tool_func(self, name: str) -> Optional[ToolFunction]:
        """根据名称获取工具的执行函数"""
        tool_obj = self._tools.get(name)
        return tool_obj.function if tool_obj else None

    def generate_tools_description(self) -> str:
        """生成供 LLM 使用的工具描述字符串"""
        if not self._tools:
            return "No tools available."

        descriptions = [
            f"- {tool.name}: {tool.description}" for tool in self._tools.values()
        ]
        return "\n".join(descriptions)

    def list_tools(self) -> list[str]:
        """返回所有工具名称列表"""
        return list(self._tools.keys())


# --- Tool 注册装饰器 ---
def tool(
    tool_system: ToolSystem, name: str, description: str
) -> Callable[[ToolFunction], ToolFunction]:
    """工具注册装饰器，将函数注册到 ToolSystem"""

    def decorator(func: ToolFunction) -> ToolFunction:
        @wraps(func)
        def wrapper(**kwargs: Any) -> ToolResult:
            # 保证工具函数始终通过 kwargs 被调用，保持函数式风格
            return func(**kwargs)

        tool_system.register_tool(
            Tool(name=name, description=description, function=wrapper)
        )
        return wrapper

    return decorator


# 全局工具系统实例，供所有工具模块导入和注册
GLOBAL_TOOLS = ToolSystem()
