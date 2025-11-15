# src/mas_homework/agent_core.py
import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

# 确保这些模块和类型可以被正确导入
from .llm_client import LLMClient
from .tool_system import GLOBAL_TOOLS, ToolResult, ToolSystem
from .types import (  # 导入 Message 以配合 LLMResponse
    AgentState,
    LLMParsedContent,
    LLMResponse,
    Message,
    StepRecord,
    ToolCall,
)

LOG_FILE = Path.cwd() / "agent.log"
HISTORY_FILE = Path.cwd() / "agent_history.json"
MAX_LOG_LENGTH = 1000  # LLM 输出日志截断长度


def write_log(message: str):
    """写入带时间戳的日志文件，并限制长度"""
    timestamp = datetime.datetime.now().isoformat()
    if len(message) > MAX_LOG_LENGTH:
        message = message[:MAX_LOG_LENGTH] + "..."
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def load_history() -> List[StepRecord]:
    """从文件加载历史记录，实现长期记忆"""
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            return [StepRecord.model_validate(d) for d in data]
        except Exception as e:
            write_log(f"Error loading history from {HISTORY_FILE}: {e}")
            return []
    return []


def save_history(history: List[StepRecord]):
    """保存历史记录到文件"""
    try:
        # 使用 model_dump 确保 Pydantic 对象被转换为字典
        HISTORY_FILE.write_text(
            json.dumps([s.model_dump() for s in history], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        write_log(f"Error saving history to {HISTORY_FILE}: {e}")


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
        self.max_consecutive_failures = 3  # 连续解析失败退出限制
        write_log(f"\n--- New Agent Session Initialized (Model: {model}) ---")

    def _format_history_for_debug(
        self, history: List[StepRecord]
    ) -> List[Dict[str, Any]]:
        """格式化历史记录用于日志输出"""
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
                obs = step.observation
                item["observation"] = obs if len(obs) < 500 else obs[:500] + "..."
            debug_history.append(item)
        write_log(
            f"Current History Summary: {json.dumps(debug_history, ensure_ascii=False, indent=2)}"
        )
        return debug_history

    def _get_system_prompt(self) -> str:
        """生成严格 ReAct 模式的系统提示词"""
        tool_desc = self.tool_system.get_tool_descriptions()
        return (
            "You are a ReAct-style document analysis agent. Each step MUST produce a single JSON object.\n"
            "STRICTLY OUTPUT ONLY JSON. NO OTHER TEXT, COMMENTS, OR MARKDOWN.\n"
            "1. 'thought': what you are thinking (MANDATORY).\n"
            "2. **MUTUALLY EXCLUSIVE**: EITHER 'tool_call' (to get new info) OR 'final_answer' (to conclude), NOT both.\n"
            "\nTool call structure:\n"
            '{"name": "...", "arguments": {...}}\n'
            "Final answer must be a string containing a complete JSON object.\n"
            "Always output your Thought even before calling a tool.\n"
            f"\nAvailable tools:\n{tool_desc}\n"
            "\nExample Tool Call:\n"
            '  {"thought": "I need to find files.", "tool_call": {"name": "ls", "arguments": {"path_or_pattern": "*.pdf"}}}\n'
            "\nExample Final Answer:\n"
            '  {"thought": "I have gathered all data.", "final_answer": "{ "summary": "Final result goes here" }"\n'
            "\nDO NOT include any 'finish' tool. Use the 'final_answer' key to conclude."
        )

    def _format_messages(self, state: AgentState) -> List[Dict[str, str]]:
        """将 AgentState 转换为 LLM 接口所需的 message 格式"""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._get_system_prompt()},
        ]

        # 构造历史记录消息 (包括上一次对话的记忆)
        for step in state.history:
            if step.thought:
                llm_content: Dict[str, Any] = {"thought": step.thought}

                # 构造 LLM (assistant) 的输出
                if step.tool_name:
                    llm_content["tool_call"] = {
                        "name": step.tool_name,
                        "arguments": step.tool_args or {},
                    }
                elif step.status == "COMPLETED" and state.final_answer:
                    # 确保只在 COMPLETED 状态且有 final_answer 时才输出
                    llm_content["final_answer"] = state.final_answer

                messages.append(
                    {"role": "assistant", "content": json.dumps(llm_content)}
                )

            # 构造工具结果 (tool_result) 的观察
            if step.observation and step.status != "COMPLETED":
                # tool_result 角色用于传递工具执行结果给 LLM
                messages.append({"role": "tool_result", "content": step.observation})

        # 当前用户查询
        messages.append({"role": "user", "content": state.user_query})
        return messages

    def _parse_llm_response(
        self, llm_response: LLMResponse
    ) -> Tuple[LLMParsedContent, Optional[str]]:
        """使用正则匹配解析 LLM 响应，提高鲁棒性，并校验互斥结构"""
        raw_content = llm_response.message.content or ""
        write_log(f"LLM Raw Output: {raw_content}")

        # 使用正则尝试提取第一个最外层 JSON 对象
        # re.DOTALL 确保匹配能跨越多行，这是解决 boundary parse fail 的关键
        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if not match:
            return LLMParsedContent(
                thought="Error"
            ), "Failed to find valid JSON boundaries."

        json_payload = match.group(0)

        try:
            data = json.loads(json_payload)
            # 依赖 types.py 中的 LLMParsedContent 校验互斥性
            parsed_content = LLMParsedContent.model_validate(data)

            # types.py 已经做了互斥校验，这里只检查 thought 是否存在
            if not parsed_content.thought:
                return parsed_content, "JSON missing mandatory 'thought' field."

            return parsed_content, None

        except json.JSONDecodeError as e:
            return LLMParsedContent(thought="Error"), f"Failed to parse JSON: {e}"
        except ValidationError as e:
            return LLMParsedContent(thought="Error"), f"Validation Error: {e}"
        except Exception as e:
            return LLMParsedContent(
                thought="Error"
            ), f"Unexpected parsing error: {type(e).__name__}: {e}"

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """执行工具调用，并增加人工确认步骤"""
        tool_name = tool_call.name
        tool_args = tool_call.arguments

        # 1. 人工确认
        print(
            f"\n[CONFIRM] Agent suggests executing tool '{tool_name}' with args {tool_args}."
        )
        confirm = input(f"Confirm execution? (y/n): ")
        if confirm.lower() != "y":
            write_log(f"Tool execution '{tool_name}' cancelled by user.")
            return ToolResult(
                success=False, output=None, error="Tool execution cancelled by user."
            )

        # 2. 实际执行
        if tool_name not in self.tool_system.tools:
            return ToolResult(
                success=False, output=None, error=f"Unknown tool: '{tool_name}'"
            )
        try:
            result = self.tool_system.run_tool(tool_name, **tool_args)
            if not isinstance(result, ToolResult):
                result = ToolResult(
                    success=False,
                    output=None,
                    error=f"Tool '{tool_name}' returned invalid structure.",
                )

            write_log(f"Tool '{tool_name}' executed. Success: {result.success}")
            return result
        except Exception as e:
            write_log(f"Tool '{tool_name}' execution failed: {type(e).__name__}: {e}")
            return ToolResult(
                success=False, output=None, error=f"Tool execution exception: {e}"
            )

    async def run(self, user_query: str) -> str:
        """Agent 主循环"""

        # 1. 加载历史并初始化状态
        history = load_history()
        # 将用户查询作为历史记录中的最新一步，并基于历史长度设置迭代次数
        state = AgentState(
            user_query=user_query, history=history, iteration=len(history)
        )
        consecutive_failures = 0

        write_log(f"Starting run loop for query: {user_query}")

        while state.iteration < self.max_steps and state.final_answer is None:
            # 迭代次数增加放在循环开始，确保 history 长度正确
            state = state.model_copy(update={"iteration": state.iteration + 1})

            # 准备消息，包含历史
            messages = self._format_messages(state)

            print(f"\n[AGENT] Step {state.iteration}. Thinking...")

            try:
                # 2. 调用 LLM
                llm_response = await self.llm_client.chat(
                    self.model, messages, format="json"
                )
            except Exception as e:
                error_obs = (
                    f"LLM client call failed: {type(e).__name__}: {str(e)[:500]}"
                )
                current_step = StepRecord(
                    iteration=state.iteration,
                    thought="LLM call failed.",
                    observation=error_obs,
                    status="FAILED",
                )
                state = state.model_copy(
                    update={"history": state.history + [current_step]}
                )
                write_log(f"LLM call exception: {error_obs}")
                break  # LLM 客户端失败，直接退出

            # 3. 解析 LLM 输出
            parsed_content, parse_error = self._parse_llm_response(llm_response)

            # 打印 LLM Thought
            print(f"[Thought] {parsed_content.thought}")

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

            if parse_error:
                # 4a. 解析失败
                consecutive_failures += 1
                current_step = current_step.model_copy(
                    update={
                        "observation": f"Parsing Error: {parse_error}",
                        "status": "FAILED",
                    }
                )
                state = state.model_copy(
                    update={"history": state.history + [current_step]}
                )
                write_log(f"Parsing error at step {state.iteration}: {parse_error}")
                if consecutive_failures >= self.max_consecutive_failures:
                    write_log(
                        f"Consecutive failures >= {self.max_consecutive_failures}. Aborting."
                    )
                    break
                continue

            consecutive_failures = 0

            if parsed_content.final_answer is not None:
                # 4b. 终结答案
                try:
                    # 确保 final_answer 是 JSON 字符串
                    json.loads(parsed_content.final_answer)
                except Exception:
                    # 如果不是 JSON 字符串，则包装，以确保最终返回格式
                    parsed_content.final_answer = json.dumps(
                        {"result": parsed_content.final_answer}
                    )

                current_step = current_step.model_copy(update={"status": "COMPLETED"})
                state = state.model_copy(
                    update={
                        "history": state.history + [current_step],
                        "final_answer": parsed_content.final_answer,
                    }
                )
                write_log(f"Final answer received at step {state.iteration}")
                break

            elif parsed_content.tool_call is not None:
                # 4c. 工具调用
                tool_result = await self._execute_tool(parsed_content.tool_call)
                observation_content = (
                    tool_result.output
                    or tool_result.error
                    or "Tool returned no output/error."
                )
                status = "SUCCESS" if tool_result.success else "FAILED"

                # 更新状态
                current_step = current_step.model_copy(
                    update={"observation": observation_content, "status": status}
                )
                state = state.model_copy(
                    update={"history": state.history + [current_step]}
                )

                print(
                    f"[Observation] Status: {status}, Output: {observation_content[:100]}..."
                )
                write_log(
                    f"Tool '{parsed_content.tool_call.name}' executed at step {state.iteration}, status: {status}"
                )

        # 5. 循环结束处理
        save_history(state.history)  # 保存最终历史

        if state.final_answer:
            print(f"\n[DONE] Task completed in {state.iteration} steps.")
            return state.final_answer

        # 异常退出或最大步数
        history_summary = self._format_history_for_debug(state.history)
        error_details = {
            "error": f"Agent reached maximum steps ({self.max_steps}) or consecutive failures ({consecutive_failures}) without final answer.",
            "last_state_history": history_summary,
        }
        write_log(f"Max steps/consecutive failures reached. Returning error JSON.")
        return json.dumps(error_details, indent=4, ensure_ascii=False)
