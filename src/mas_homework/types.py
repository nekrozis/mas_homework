# src/mas_homework/types.py
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, ValidationError

# --- 工具与结果类型 ---

# LLM 输出的工具调用结构
class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

# 工具函数统一返回的结构 (仍使用 TypedDict 保持 FP 风格的简洁性)
class ToolResult(Dict):
    success: bool
    output: Optional[str]
    error: Optional[str]

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
    
    class Config:
        # Pydantic V1/V2 兼容性设置
        frozen = True # 确保 AgentState 实例是不可变的

# --- LLM 响应类型 ---
# 这是一个简化的 Ollama/OpenAI 响应结构
class MessageContent(BaseModel):
    content: str

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None # 理论上LLM会返回ToolCall

class LLMResponse(BaseModel):
    """LLM 原始响应的结构 (Pydantic Model)"""
    message: Message
    # Ollama response usually contains 'model', 'created_at', etc.
    model: Optional[str] = None
    
# LLM 解析后的内容结构 (用于从LLM的content字符串中解析出的JSON)
class LLMParsedContent(BaseModel):
    """LLM 输出的 JSON 结构 (Thought + Action/Finish)"""
    thought: str
    tool_call: Optional[ToolCall] = None
    final_answer: Optional[str] = None