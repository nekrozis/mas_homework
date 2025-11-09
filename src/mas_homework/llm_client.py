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

    async def chat(
        self, model: str, messages: List[Dict[str, str]], format: Optional[str] = None
    ) -> LLMResponse:
        options = {"format": format} if format else {}

        # 必须在这里初始化，以便在 except 块中引用其内容
        raw_response: Any = None

        try:
            # 1. 获取原始响应 (Pylance 接受赋值给 Any)
            raw_response = await self._client.chat(
                model=model, messages=messages, options=options
            )

            # 2. 使用 Pydantic 进行严格验证和类型转换
            return LLMResponse.model_validate(raw_response)

        except ValidationError as e:
            # 此时 raw_response 必然已被赋值
            raw_response_str = json.dumps(raw_response, indent=2)
            raise RuntimeError(
                f"LLM response validation failed: {e}\nRaw Response: {raw_response_str}"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama client error: {e}")
