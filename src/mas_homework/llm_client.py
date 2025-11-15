# src/mas_homework/llm_client.py
import json
from typing import Any, Dict, List, Optional

import ollama
from pydantic import ValidationError

from .types import LLMResponse


class LLMClient:
    async def chat(
        self, model: str, messages: List[Dict[str, str]], format: Optional[str] = None
    ) -> LLMResponse:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self):
        self._client = ollama.AsyncClient()

    def _to_json_serializable(self, obj: Any) -> Dict[str, Any]:
        """将响应对象转换为 JSON 可序列化字典"""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, dict):
            return obj
        try:
            return dict(obj)
        except:
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
            # 获取 Ollama 原始响应
            raw_response = await self._client.chat(
                model=model, messages=messages, options=options
            )

            # 转换为可序列化字典以便 Pydantic 校验
            serializable_response = self._to_json_serializable(raw_response)

            # 使用 Pydantic 验证并返回
            return LLMResponse.model_validate(serializable_response)

        except ValidationError as e:
            serializable_response = self._to_json_serializable(raw_response)
            raw_response_str = json.dumps(
                serializable_response, indent=2, ensure_ascii=False
            )
            raise RuntimeError(
                f"LLM response validation failed: {e}\nRaw Response: {raw_response_str}"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama client error: {e}")
