# src/mas_homework/llm_client.py
import json
from typing import Any, Dict, List, Optional

import ollama
from pydantic import ValidationError

from .types import LLMResponse  # 假设你的 types.py 中定义了 LLMResponse


class LLMClient:
    async def chat(
        self, model: str, messages: List[Dict[str, str]], format: Optional[str] = None
    ) -> LLMResponse:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self):
        self._client = ollama.AsyncClient()

    def _to_json_serializable(self, obj: Any) -> Dict[str, Any]:
        """将复杂的响应对象转换为 JSON 可序列化的字典。"""
        # 尝试使用 Pydantic 的 model_dump 方法
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, dict):
            return obj
        # 针对 ollama 客户端返回的响应对象，通常可以转换为字典
        try:
            # 这通常适用于 ollama.ChatResponse
            return dict(obj)
        except:
            # 如果以上都失败，尝试转换为 __dict__ 或返回错误
            return getattr(
                obj,
                "__dict__",
                {"error": f"Cannot serialize object type {type(obj).__name__}"},
            )

    async def chat(
        self, model: str, messages: List[Dict[str, str]], format: Optional[str] = None
    ) -> LLMResponse:
        options = {"format": format} if format else {}

        raw_response: Any = None

        try:
            # 1. 获取原始响应 (ollama.ChatResponse 对象)
            raw_response = await self._client.chat(
                model=model, messages=messages, options=options
            )

            # === 修复点：在 Pydantic 验证前，将 Ollama 响应对象转换为字典 ===
            serializable_response = self._to_json_serializable(raw_response)

            # 2. 使用 Pydantic 进行严格验证和类型转换
            return LLMResponse.model_validate(serializable_response)
            # =============================================================

        except ValidationError as e:
            # 此时 raw_response 必然已被赋值。使用安全序列化方法进行调试输出。
            serializable_response = self._to_json_serializable(raw_response)
            raw_response_str = json.dumps(
                serializable_response, indent=2, ensure_ascii=False
            )

            raise RuntimeError(
                f"LLM response validation failed: {e}\nRaw Response: {raw_response_str}"
            )
        except Exception as e:
            # 保留对其他所有 ollama 错误的通用处理
            raise RuntimeError(f"Ollama client error: {e}")
