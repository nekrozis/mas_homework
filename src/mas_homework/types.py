# src/mas_homework/types.py
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

# --- 工具相关类型 ---


class ToolCall(BaseModel):
    """LLM 输出的工具调用结构"""

    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """工具执行结果，保证结构化和类型安全"""

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
    """Agent 不可变状态"""

    user_query: str
    history: List[StepRecord] = Field(default_factory=list)
    iteration: int = 0
    final_answer: Optional[str] = None
    error: Optional[str] = None  # 用于记录异常或错误信息

    model_config = {
        "frozen": True  # 确保实例不可变
    }


# --- LLM 响应类型 ---


class Message(BaseModel):
    """LLM 消息结构"""

    role: str
    content: Optional[str] = None


class LLMResponse(BaseModel):
    """LLM 原始响应"""

    message: Message
    model: Optional[str] = None


class LLMParsedContent(BaseModel):
    """解析后的 LLM 输出 JSON 结构"""

    thought: str
    tool_call: Optional[ToolCall] = None
    final_answer: Optional[str] = None

    @model_validator(mode="after")
    def check_mutual_exclusivity(self) -> "LLMParsedContent":
        """确保 tool_call 和 final_answer 互斥，除非 thought='Error'"""
        has_tool = self.tool_call is not None
        has_final = self.final_answer is not None

        if has_tool and has_final:
            raise ValueError(
                "LLM output must contain EITHER 'tool_call' OR 'final_answer', not both."
            )
        if not has_tool and not has_final and self.thought != "Error":
            raise ValueError(
                "LLM output must contain EITHER 'tool_call' OR 'final_answer'."
            )

        return self
