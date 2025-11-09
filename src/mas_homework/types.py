# types.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

# --- 工具与结果类型 ---


class ToolCall(TypedDict):
    """LLM 输出的工具调用结构"""

    name: str
    arguments: Dict[str, Any]


class ToolResult(TypedDict):
    """工具函数统一返回的结构"""

    success: bool
    output: Optional[str]
    error: Optional[str]


# --- Agent 状态与历史记录类型 ---


@dataclass(frozen=True, kw_only=True)
class StepRecord:
    """Agent 每一步的结构化记录"""

    iteration: int
    thought: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    status: str = "PENDING"  # SUCCESS, TOOL_ERROR, ACTION_ERROR, FINISHED


@dataclass(frozen=True, kw_only=True)
class AgentState:
    """Agent 的不可变状态"""

    user_query: str
    history: List[StepRecord] = field(default_factory=list)
    iteration: int = 0
    final_answer: Optional[str] = None


# --- LLM 响应类型 ---
# LLM 客户端的 chat 方法返回的简化结构
LLMResponse = Dict[str, Any]

# LLM 响应中解析出的内容（可以是 ToolCall 或 FinalAnswer）
LLMParsedContent = Dict[str, Any]
