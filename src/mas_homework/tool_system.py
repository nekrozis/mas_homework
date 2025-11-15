# src/mas_homework/tool_system.py
import inspect
from typing import Any, Callable, Dict, TypeVar

from .types import ToolResult  # 假设 ToolResult 是 TypedDict 或 pydantic.BaseModel

F = TypeVar("F", bound=Callable[..., ToolResult])


class Tool:
    def __init__(self, name: str, description: str, func: Callable[..., ToolResult]):
        self.name = name
        self.description = description
        self.func = func
        self.signature = inspect.signature(func)

    def generate_prompt_description(self) -> str:
        """生成工具描述，用于系统提示词"""
        params = []
        for p in self.signature.parameters.values():
            if p.annotation != inspect.Parameter.empty:
                # 使用类型名，如果注解是 typing 类型则取 __name__ 会报错，fallback to str
                try:
                    type_name = p.annotation.__name__
                except AttributeError:
                    type_name = str(p.annotation)
            else:
                type_name = "Any"
            params.append(f"{p.name}: {type_name}")
        param_str = ", ".join(params)
        return f"{self.name}({param_str}): {self.description}"


class ToolSystem:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def add_tool(self, tool: Tool) -> None:
        """注册工具"""
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' already exists.")
        self.tools[tool.name] = tool

    def get_tool_descriptions(self) -> str:
        """返回所有工具描述，用于系统提示词"""
        return "\n".join(
            tool.generate_prompt_description() for tool in self.tools.values()
        )

    def run_tool(self, name: str, **kwargs: Any) -> ToolResult:
        """执行指定工具"""
        if name not in self.tools:
            return ToolResult(
                success=False, output=None, error=f"Tool '{name}' not found."
            )
        tool_obj = self.tools[name]
        return tool_obj.func(**kwargs)


def tool(system: ToolSystem, name: str, description: str) -> Callable[[F], F]:
    """装饰器形式注册工具"""

    def decorator(func: F) -> F:
        system.add_tool(Tool(name, description, func))
        return func

    return decorator


# 全局工具系统
GLOBAL_TOOLS = ToolSystem()
