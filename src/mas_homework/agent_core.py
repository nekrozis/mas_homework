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

    def _format_messages(self, state: AgentState) -> List[Dict[str, str]]:
        """构建发送给 LLM 的消息历史"""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": state.user_query},
        ]

        # 将历史记录中的 Thought, Action, Observation 转换为 LLM 格式
        for step in state.history:
            if step.thought:
                # 构建 LLM 响应的 JSON 内容
                llm_content: Dict[str, Any] = {"thought": step.thought}
                if step.tool_name:
                    llm_content["tool_call"] = {
                        "name": step.tool_name,
                        "arguments": step.tool_args or {},
                    }
                elif step.status == "COMPLETED":
                    # 在最终答案步骤，LLM的输出是 Thought + Final Answer
                    llm_content["final_answer"] = state.final_answer or ""

                # LLM 响应：将复杂字典转为 JSON 字符串，满足 List[Dict[str, str]] 要求
                messages.append(
                    {"role": "assistant", "content": json.dumps(llm_content)}
                )

            if step.observation:
                # 工具执行结果 (Observation)
                messages.append({"role": "tool_result", "content": step.observation})

        return messages

    def _get_system_prompt(self) -> str:
        """生成系统提示词，包含工具说明"""
        # 依赖 tool_system.get_tool_descriptions() (已在 tool_system.py 中修复)
        tool_desc = self.tool_system.get_tool_descriptions()
        return (
            "You are a sophisticated document analysis assistant. Your task is to process a PDF file "
            "based on the user's query and provide a structured JSON response in the FINAL_ANSWER format.\n\n"
            "You MUST follow the ReAct pattern: Thought, then Action (tool_call) OR Final Answer (final_answer).\n"
            "Your output MUST be a single JSON object containing 'thought' and EITHER 'tool_call' OR 'final_answer'.\n\n"
            "Tool Descriptions:\n"
            f"{tool_desc}\n\n"
            "JSON Output Format Requirements:\n"
            "1. Tool Call Example:\n"
            '   {{"thought": "I need to find files.", "tool_call": {{"name": "ls", "arguments": {{"path_or_pattern": "*.pdf"}} }}}}\n'
            "2. Final Answer Example (MUST contain the final JSON structure for the user query):\n"
            '   {{"thought": "I have gathered all data.", "final_answer": "{{ ... final JSON analysis ... }}"}}\n'
            "The final_answer value must be a string containing a complete JSON object."
        )

    def _parse_llm_response(
        self, llm_response: LLMResponse
    ) -> Tuple[LLMParsedContent, Optional[str]]:
        """解析 LLM 的 JSON 响应，并返回解析后的内容和错误（如果有）"""
        raw_content = llm_response.message.content
        if not raw_content:
            return LLMParsedContent(thought="Error"), "LLM returned empty content."

        try:
            data = json.loads(raw_content)

            # 使用 Pydantic 严格验证 LLM 的 JSON 结构
            parsed_content = LLMParsedContent.model_validate(data)

            # 额外的逻辑检查
            has_tool = parsed_content.tool_call is not None
            has_final = parsed_content.final_answer is not None

            if has_tool and has_final:
                return (
                    parsed_content,
                    "JSON contains both 'tool_call' and 'final_answer'. Use only one.",
                )

            if not has_tool and not has_final:
                return (
                    parsed_content,
                    "JSON contains neither 'tool_call' nor 'final_answer'.",
                )

            return parsed_content, None

        except json.JSONDecodeError:
            return LLMParsedContent(
                thought="Error"
            ), f"Failed to decode JSON from LLM: {raw_content[:100]}..."
        except ValidationError as e:
            return LLMParsedContent(
                thought="Error"
            ), f"LLM JSON structure validation failed: {e}"
        except Exception as e:
            return LLMParsedContent(thought="Error"), f"Unexpected parsing error: {e}"

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """执行工具并返回结果"""
        tool_name = tool_call.name
        tool_args = tool_call.arguments

        # 依赖 tool_system.tools 属性和 run_tool 方法 (已在 tool_system.py 中修复)

        if tool_name not in self.tool_system.tools:
            return ToolResult(
                success=False, output=None, error=f"Unknown tool: '{tool_name}'"
            )

        # 执行工具
        try:
            return self.tool_system.run_tool(tool_name, **tool_args)
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{tool_name}' failed with exception: {type(e).__name__}: {e}",
            )

    async def run(self, user_query: str) -> str:
        """运行 ReAct 循环直到完成或达到最大步数"""

        # 使用不可变 AgentState 初始化状态
        state = AgentState(user_query=user_query)

        while state.iteration < self.max_steps and state.final_answer is None:
            # 1. 构造消息并调用 LLM
            messages = self._format_messages(state)
            try:
                llm_response = await self.llm_client.chat(
                    self.model, messages, format="json"
                )
            except Exception as e:
                # 记录 LLM 调用失败
                error_obs = f"LLM client call failed: {e}"
                # 使用 Pydantic 构造新的不可变状态
                new_history = state.history + [
                    StepRecord(
                        iteration=state.iteration,
                        thought="Attempting to call LLM failed.",
                        observation=error_obs,
                        status="FAILED",
                    )
                ]
                state = AgentState(
                    user_query=state.user_query,
                    history=new_history,
                    iteration=state.iteration + 1,
                )
                continue

            # 2. 解析 LLM 响应
            parsed_content, parse_error = self._parse_llm_response(llm_response)

            # 创建当前步骤的记录 (需要先初始化，才能在下面更新 observation/status)
            current_step = StepRecord(
                iteration=state.iteration,
                thought=parsed_content.thought,
                tool_name=parsed_content.tool_call.name
                if parsed_content.tool_call
                else None,
                tool_args=parsed_content.tool_call.arguments
                if parsed_content.tool_call
                else None,
            )

            # 3. 处理解析错误
            if parse_error:
                current_step = StepRecord(
                    **current_step.model_dump(),
                    observation=f"Parsing Error: {parse_error}",
                    status="FAILED",
                )

                # LLM 输出格式错误，将错误作为 Observation 反馈给 LLM
                new_history = state.history + [current_step]
                state = AgentState(
                    user_query=state.user_query,
                    history=new_history,
                    iteration=state.iteration + 1,
                )
                continue

            # 4. 执行工具 或 确认终结
            if parsed_content.final_answer is not None:
                # 4a. 终结：LLM 给出最终答案 (LLM 绕过了 finish 工具)
                current_step = StepRecord(
                    **current_step.model_dump(), status="COMPLETED"
                )
                new_history = state.history + [current_step]
                # 更新状态并返回最终答案
                state = AgentState(
                    user_query=state.user_query,
                    history=new_history,
                    iteration=state.iteration + 1,
                    final_answer=parsed_content.final_answer,
                )
                break

            elif parsed_content.tool_call is not None:
                # 4b. 工具调用：执行工具
                tool_result = await self._execute_tool(parsed_content.tool_call)

                observation_content = (
                    tool_result.output
                    or tool_result.error
                    or "Tool returned no output/error."
                )
                status = "SUCCESS" if tool_result.success else "FAILED"

                # 检查特殊终结工具 'finish' 的结果
                if tool_result.output and tool_result.output.startswith(
                    "TASK_COMPLETED:"
                ):
                    final_answer = tool_result.output.replace("TASK_COMPLETED:", "", 1)
                    status = "COMPLETED"

                    current_step = StepRecord(
                        **current_step.model_dump(),
                        observation=observation_content,
                        status=status,
                    )
                    new_history = state.history + [current_step]
                    # finish 工具执行成功，返回最终答案
                    state = AgentState(
                        user_query=state.user_query,
                        history=new_history,
                        iteration=state.iteration + 1,
                        final_answer=final_answer,
                    )
                    break

                # 普通工具执行成功或失败，将 Observation 反馈给 LLM
                current_step = StepRecord(
                    **current_step.model_dump(),
                    observation=observation_content,
                    status=status,
                )
                new_history = state.history + [current_step]
                state = AgentState(
                    user_query=state.user_query,
                    history=new_history,
                    iteration=state.iteration + 1,
                )

        # 循环结束，返回结果
        if state.final_answer:
            return state.final_answer
        elif state.iteration >= self.max_steps:
            return json.dumps(
                {
                    "error": f"Agent reached maximum steps ({self.max_steps}) without providing a final answer."
                }
            )
        else:
            return json.dumps({"error": "Agent loop terminated unexpectedly."})
