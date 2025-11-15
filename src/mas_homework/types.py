# src/mas_homework/types.py
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

# --- 工具与结果类型 ---


# LLM 输出的工具调用结构 (保持不变)
class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)  # 增加默认值以提高健壮性


# **关键修改：ToolResult 必须是 BaseModel，以匹配 AgentCore 中的类型检查和结构化返回**
class ToolResult(BaseModel):
    """表示工具执行的结果，确保结构化和类型安全"""

    success: bool
    output: Optional[str] = None
    error: Optional[str] = None


# --- Agent 状态与历史记录类型 ---


class StepRecord(BaseModel):
    """Agent 每一步的结构化记录"""

    iteration: int
    thought: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    status: str = "PENDING"


class AgentState(BaseModel):
    """Agent 的不可变状态"""

    user_query: str
    history: List[StepRecord] = Field(default_factory=list)
    iteration: int = 0
    final_answer: Optional[str] = None

    model_config = {
        "frozen": True  # 确保 AgentState 实例是不可变的 (Pydantic V2 风格)
    }


# --- LLM 响应类型 ---


class Message(BaseModel):
    """LLM 消息中的内容和角色"""

    role: str
    content: Optional[str] = None
    # 移除 tool_calls，因为我们依赖 content 字符串中的 JSON 解析 ReAct 结构
    # tool_calls: Optional[List[ToolCall]] = None


class LLMResponse(BaseModel):
    """LLM 原始响应的结构 (Pydantic Model)"""

    message: Message
    model: Optional[str] = None
    # 理论上还会有 created_at, usage 等，这里简化


# LLM 解析后的内容结构 (用于从LLM的content字符串中解析出的JSON)
class LLMParsedContent(BaseModel):
    """LLM 输出的 JSON 结构 (Thought + Action/Finish)"""

    thought: str
    tool_call: Optional[ToolCall] = None
    final_answer: Optional[str] = None

    @model_validator(mode="after")
    def check_mutual_exclusivity(self) -> "LLMParsedContent":
        """确保 tool_call 和 final_answer 互斥，且至少存在一个 (除非 thought='Error')"""
        has_tool = self.tool_call is not None
        has_final = self.final_answer is not None

        # 1. 严格校验：两者必须互斥
        if has_tool and has_final:
            raise ValueError(
                "LLM output must contain EITHER 'tool_call' OR 'final_answer', not both."
            )

        # 2. 校验：至少存在一个动作，除非是解析失败时的 Thought="Error"
        if not has_tool and not has_final and self.thought != "Error":
            raise ValueError(
                "LLM output must contain EITHER 'tool_call' OR 'final_answer'."
            )

        # 3. 校验 final_answer 是否看起来像 JSON 字符串 (可选，但推荐)
        if has_final and has_final is not None:
            # 如果不是 None，尝试检查是否为 JSON 字符串，如果不是，AgentCore 运行时会尝试包裹
            # 这里不做严格校验，留给 AgentCore 处理，以提高 LLM 的容错空间。
            pass

        return self
