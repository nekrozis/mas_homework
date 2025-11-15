# src/mas_homework/agent_core.py
import json
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from .llm_client import LLMClient
from .tool_system import GLOBAL_TOOLS, ToolResult, ToolSystem
from .types import AgentState, LLMParsedContent, LLMResponse, StepRecord, ToolCall


class Agent:
    def __init__(
        self,
        llm_client: LLMClient,
        tool_system: ToolSystem = GLOBAL_TOOLS,
        max_steps: int = 10,
        model: str = "llama3",
    ):
        self.llm_client = llm_client
        self.tool_system = tool_system
        self.max_steps = max_steps
        self.model = model

    def _format_history_for_debug(
        self, history: List[StepRecord]
    ) -> List[Dict[str, Any]]:
        """格式化历史记录，便于调试"""
        debug_history = []
        for step in history:
            item: Dict[str, Any] = {
                "step": step.iteration,
                "thought": step.thought,
                "status": step.status,
            }
            if step.tool_name:
                item["action"] = step.tool_name
                item["args"] = step.tool_args
            if step.observation:
                item["observation"] = (
                    step.observation[:500] + "..."
                    if len(step.observation) > 500
                    else step.observation
                )
            debug_history.append(item)
        return debug_history

    def _format_messages(self, state: AgentState) -> List[Dict[str, str]]:
        """构建发送给 LLM 的消息列表"""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": state.user_query},
        ]
        for step in state.history:
            if step.thought:
                llm_content: Dict[str, Any] = {"thought": step.thought}
                if step.tool_name:
                    llm_content["tool_call"] = {
                        "name": step.tool_name,
                        "arguments": step.tool_args or {},
                    }
                elif step.status == "COMPLETED":
                    llm_content["final_answer"] = state.final_answer or ""
                messages.append(
                    {"role": "assistant", "content": json.dumps(llm_content)}
                )
            if step.observation:
                messages.append({"role": "tool_result", "content": step.observation})
        return messages

    def _get_system_prompt(self) -> str:
        tool_desc = self.tool_system.get_tool_descriptions()
        return (
            "You are a sophisticated document analysis assistant. Your task is to process a PDF file "
            "based on the user's query and provide a structured JSON response in the FINAL_ANSWER format.\n\n"
            "Follow the ReAct pattern: Thought, then Action (tool_call) OR Final Answer (final_answer).\n"
            "Your output MUST be a single JSON object containing 'thought' and EITHER 'tool_call' OR 'final_answer'.\n\n"
            f"Tool Descriptions:\n{tool_desc}\n\n"
            "JSON Output Format Requirements:\n"
            'Tool Call Example: {"thought": "I need to find files.", "tool_call": {"name": "ls", "arguments": {"path_or_pattern": "*.pdf"} }}\n'
            'Final Answer Example: {"thought": "I have gathered all data.", "final_answer": "{ ... final JSON analysis ... }"}\n'
            "The final_answer value must be a string containing a complete JSON object."
        )

    def _parse_llm_response(
        self, llm_response: LLMResponse
    ) -> Tuple[LLMParsedContent, Optional[str]]:
        """解析 LLM JSON 响应"""
        raw_content = llm_response.message.content or ""
        start_index = raw_content.find("{")
        end_index = raw_content.rfind("}")
        if start_index == -1 or end_index == -1 or start_index > end_index:
            return LLMParsedContent(
                thought="Error"
            ), f"Failed to find JSON boundaries in LLM response: {raw_content[:100]}..."
        json_payload = raw_content[start_index : end_index + 1]
        try:
            data = json.loads(json_payload)
            parsed = LLMParsedContent.model_validate(data)
            has_tool, has_final = (
                parsed.tool_call is not None,
                parsed.final_answer is not None,
            )
            if has_tool and has_final:
                return parsed, "JSON contains both 'tool_call' and 'final_answer'."
            if not has_tool and not has_final:
                return parsed, "JSON contains neither 'tool_call' nor 'final_answer'."
            return parsed, None
        except (json.JSONDecodeError, ValidationError) as e:
            return LLMParsedContent(
                thought="Error"
            ), f"JSON parsing/validation failed: {e}"
        except Exception as e:
            return LLMParsedContent(thought="Error"), f"Unexpected parsing error: {e}"

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """执行工具"""
        tool_name, tool_args = tool_call.name, tool_call.arguments
        if tool_name not in self.tool_system.tools:
            return ToolResult(
                success=False, output=None, error=f"Unknown tool: {tool_name}"
            )
        try:
            return self.tool_system.run_tool(tool_name, **tool_args)
        except Exception as e:
            return ToolResult(
                success=False, output=None, error=f"Tool '{tool_name}' failed: {e}"
            )

    def _update_state(
        self, state: AgentState, step: StepRecord, final_answer: Optional[str] = None
    ) -> AgentState:
        """统一更新 Agent 状态"""
        return AgentState(
            user_query=state.user_query,
            history=state.history + [step],
            iteration=state.iteration + 1,
            final_answer=final_answer or state.final_answer,
        )

    async def _handle_tool_result(
        self, state: AgentState, step: StepRecord, tool_call: ToolCall
    ) -> AgentState:
        """执行工具并处理 finish 工具终结逻辑"""
        result = await self._execute_tool(tool_call)
        observation = result.output or result.error or "Tool returned no output/error."
        status = "SUCCESS"
        final_answer = None

        if result.output and result.output.startswith("TASK_COMPLETED:"):
            final_answer = result.output.replace("TASK_COMPLETED:", "", 1)
            status = "COMPLETED"

        step = step.model_copy(update={"observation": observation, "status": status})
        return self._update_state(state, step, final_answer=final_answer)

    async def run(self, user_query: str) -> str:
        state = AgentState(user_query=user_query)

        while state.iteration < self.max_steps and state.final_answer is None:
            # 1. 调用 LLM
            messages = self._format_messages(state)
            try:
                llm_response = await self.llm_client.chat(
                    self.model, messages, format="json"
                )
            except Exception as e:
                step = StepRecord(
                    iteration=state.iteration,
                    thought="Attempting to call LLM failed.",
                    observation=f"LLM client call failed: {e}",
                    status="FAILED",
                )
                state = self._update_state(state, step)
                continue

            # 2. 解析 LLM 响应
            parsed, parse_error = self._parse_llm_response(llm_response)
            step = StepRecord(
                iteration=state.iteration,
                thought=parsed.thought,
                tool_name=parsed.tool_call.name if parsed.tool_call else None,
                tool_args=parsed.tool_call.arguments if parsed.tool_call else None,
            )

            if parse_error:
                step = step.model_copy(
                    update={
                        "observation": f"Parsing Error: {parse_error}",
                        "status": "FAILED",
                    }
                )
                state = self._update_state(state, step)
                continue

            # 3. 处理 LLM 输出
            if parsed.final_answer is not None:
                step = step.model_copy(update={"status": "COMPLETED"})
                state = self._update_state(
                    state, step, final_answer=parsed.final_answer
                )
                break

            if parsed.tool_call is not None:
                state = await self._handle_tool_result(state, step, parsed.tool_call)

                if state.final_answer is not None:
                    break

        if state.final_answer:
            return state.final_answer

        # 超过最大步数未完成
        return json.dumps(
            {
                "error": f"Agent reached maximum steps ({self.max_steps}) without providing a final answer.",
                "last_state_history": self._format_history_for_debug(state.history),
            },
            indent=4,
            ensure_ascii=False,
        )
