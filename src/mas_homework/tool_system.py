# src/mas_homework/tool_system.py
import inspect
from typing import Any, Callable, Dict

from .types import ToolResult  # 假设 ToolResult 是 TypedDict


class Tool:
    def __init__(self, name: str, description: str, func: Callable[..., ToolResult]):
        self.name = name
        self.description = description
        self.func = func
        self.signature = inspect.signature(func)

    def generate_prompt_description(self) -> str:
        params = [
            f"{p.name}: {p.annotation.__name__ if p.annotation != inspect.Parameter.empty else 'Any'}"
            for p in self.signature.parameters.values()
        ]
        param_str = ", ".join(params)
        return f"{self.name}({param_str}): {self.description}"


class ToolSystem:
    def __init__(self):
        # 声明 tools 属性类型
        self.tools: Dict[str, Tool] = {}

    # 添加 get_tool_descriptions 方法
    def get_tool_descriptions(self) -> str:
        """返回所有工具的描述，用于系统提示词"""
        return "\n".join(
            tool.generate_prompt_description() for tool in self.tools.values()
        )

    # 添加 run_tool 方法
    def run_tool(self, name: str, **kwargs: Any) -> ToolResult:
        """运行工具"""
        if name not in self.tools:
            return ToolResult(
                success=False, output=None, error=f"Tool '{name}' not found."
            )

        tool_obj = self.tools[name]
        return tool_obj.func(**kwargs)


def tool(system: ToolSystem, name: str, description: str):
    def decorator(func: Callable[..., ToolResult]):
        system.tools[name] = Tool(name, description, func)
        return func

    return decorator


GLOBAL_TOOLS = ToolSystem()
