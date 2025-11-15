# src/mas_homework/tool_system.py
import inspect
from typing import Any, Callable, Dict, TypeVar, get_args, get_origin

from .types import ToolResult

F = TypeVar("F", bound=Callable[..., ToolResult])


def _get_type_name(annotation: Any) -> str:
    """从类型注解中提取可读名称，支持 Optional, List 等泛型"""
    if annotation == inspect.Parameter.empty:
        return "Any"

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is None:
        return getattr(annotation, "__name__", str(annotation))

    # Optional[T]
    if origin is object.__class__.__or__ and len(args) == 2 and type(None) in args:
        inner_type = [arg for arg in args if arg is not type(None)][0]
        return f"Optional[{_get_type_name(inner_type)}]"

    # 处理 List, Dict 等泛型
    if args:
        arg_names = [_get_type_name(arg) for arg in args]
        origin_name = getattr(origin, "__name__", str(origin))
        return f"{origin_name}[{', '.join(arg_names)}]"

    return getattr(origin, "__name__", str(origin))


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
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                params.append("**kwargs: Any")
                continue

            type_name = _get_type_name(p.annotation)
            param_str = f"{p.name}: {type_name}"
            if p.default != inspect.Parameter.empty:
                param_str += " = ..."
            params.append(param_str)

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
