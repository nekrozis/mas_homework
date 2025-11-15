# src/mas_homework/agent_core.py
import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from .llm_client import LLMClient
from .tool_system import GLOBAL_TOOLS, ToolResult, ToolSystem
from .types import (
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
    """写入带时间戳的日志，并限制长度"""
    timestamp = datetime.datetime.now().isoformat()
    if len(message) > MAX_LOG_LENGTH:
        message = message[:MAX_LOG_LENGTH] + "..."
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def load_history() -> List[StepRecord]:
    """从文件加载历史记录"""
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            return [StepRecord.model_validate(d) for d in data]
        except Exception as e:
            write_log(f"Error loading history: {e}")
            return []
    return []


def save_history(history: List[StepRecord]):
    """保存历史记录到文件"""
    try:
        HISTORY_FILE.write_text(
            json.dumps([s.model_dump() for s in history], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        write_log(f"Error saving history: {e}")


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
        self.max_consecutive_failures = 3
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
        """生成严格 ReAct 模式的系统提示"""
        tool_desc = self.tool_system.get_tool_descriptions()
        return (
            "You are a ReAct-style document analysis agent. Each step MUST produce a single JSON object.\n"
            "STRICTLY OUTPUT ONLY JSON.\n"
            "1. 'thought': mandatory.\n"
            "2. MUTUALLY EXCLUSIVE: EITHER 'tool_call' OR 'final_answer'.\n"
            f"\nAvailable tools:\n{tool_desc}\n"
            "\nExample Tool Call:\n"
            '  {"thought": "I need to find files.", "tool_call": {"name": "ls", "arguments": {"path_or_pattern": "*.pdf"}}}\n'
            "\nExample Final Answer:\n"
            '  {"thought": "I have gathered all data.", "final_answer": "{ "summary": "Final result goes here" }"}\n'
            "Use 'final_answer' to conclude."
        )

    def _format_messages(self, state: AgentState) -> List[Dict[str, str]]:
        """将 AgentState 转换为 LLM 接口消息格式"""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._get_system_prompt()},
        ]

        for step in state.history:
            if step.thought:
                llm_content: Dict[str, Any] = {"thought": step.thought}
                if step.tool_name:
                    llm_content["tool_call"] = {
                        "name": step.tool_name,
                        "arguments": step.tool_args or {},
                    }
                elif step.status == "COMPLETED" and state.final_answer:
                    llm_content["final_answer"] = state.final_answer
                messages.append(
                    {"role": "assistant", "content": json.dumps(llm_content)}
                )

            if step.observation and step.status != "COMPLETED":
                observation_message = (
                    f"Tool Observation (Status: {step.status}, Tool: {step.tool_name}):\n"
                    f"{step.observation}"
                )
                messages.append({"role": "user", "content": observation_message})

        messages.append({"role": "user", "content": state.user_query})
        return messages

    def _parse_llm_response(
        self, llm_response: LLMResponse
    ) -> Tuple[LLMParsedContent, Optional[str]]:
        """解析 LLM 响应，提取 JSON 并校验互斥字段"""
        raw_content = llm_response.message.content or ""
        write_log(f"LLM Raw Output: {raw_content}")

        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if not match:
            return LLMParsedContent(thought="Error"), "Failed to find valid JSON."

        json_payload = match.group(0)
        try:
            data = json.loads(json_payload)
            parsed_content = LLMParsedContent.model_validate(data)
            if not parsed_content.thought:
                return parsed_content, "Missing mandatory 'thought' field."
            return parsed_content, None
        except json.JSONDecodeError as e:
            return LLMParsedContent(thought="Error"), f"JSON parse failed: {e}"
        except ValidationError as e:
            return LLMParsedContent(thought="Error"), f"Validation error: {e}"
        except Exception as e:
            return LLMParsedContent(
                thought="Error"
            ), f"Unexpected error: {type(e).__name__}: {e}"

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """执行工具调用并确认"""
        tool_name = tool_call.name
        tool_args = tool_call.arguments

        print(f"\n[CONFIRM] Execute tool '{tool_name}' with args {tool_args}?")
        confirm = input("Confirm? (y/n): ")
        if confirm.lower() != "y":
            write_log(f"Tool '{tool_name}' cancelled by user.")
            return ToolResult(success=False, output=None, error="Cancelled by user.")

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
                    error=f"Invalid return from '{tool_name}'",
                )
            write_log(f"Tool '{tool_name}' executed. Success: {result.success}")
            return result
        except Exception as e:
            write_log(f"Tool '{tool_name}' execution failed: {type(e).__name__}: {e}")
            return ToolResult(
                success=False, output=None, error=f"Execution exception: {e}"
            )

    async def run(self, user_query: str) -> str:
        """Agent 主循环"""
        history = load_history()
        state = AgentState(
            user_query=user_query, history=history, iteration=len(history)
        )
        consecutive_failures = 0

        write_log(f"Starting run loop for query: {user_query}")

        while state.iteration < self.max_steps and state.final_answer is None:
            state = state.model_copy(update={"iteration": state.iteration + 1})

            # 检测工具调用循环，避免重复执行
            if state.iteration > 2 and len(state.history) >= 2:
                last_entry = state.history[-1]
                if last_entry.status == "SUCCESS" and last_entry.tool_name:
                    prev_entry = state.history[-2]
                    if (
                        last_entry.tool_name == prev_entry.tool_name
                        and last_entry.tool_args == prev_entry.tool_args
                    ):
                        state = state.model_copy(
                            update={
                                "error": f"Repeated tool call '{last_entry.tool_name}' detected."
                            }
                        )
                        write_log(
                            f"LOOP DETECTED: '{last_entry.tool_name}' repeated. Aborting."
                        )
                        break

            messages = self._format_messages(state)
            print(f"\n[AGENT] Step {state.iteration}. Thinking...")

            try:
                llm_response = await self.llm_client.chat(
                    self.model, messages, format="json"
                )
            except Exception as e:
                error_obs = f"LLM call failed: {type(e).__name__}: {str(e)[:500]}"
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
                break

            parsed_content, parse_error = self._parse_llm_response(llm_response)
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
                try:
                    json.loads(parsed_content.final_answer)
                except Exception:
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
                tool_result = await self._execute_tool(parsed_content.tool_call)
                observation_content = (
                    tool_result.output or tool_result.error or "No output/error."
                )
                status = "SUCCESS" if tool_result.success else "FAILED"
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

        save_history(state.history)

        if state.final_answer:
            print(f"\n[DONE] Task completed in {state.iteration} steps.")
            return state.final_answer

        history_summary = self._format_history_for_debug(state.history)
        error_message = (
            state.error
            or f"Max steps ({self.max_steps}) or consecutive failures ({consecutive_failures}) reached without final answer."
        )
        error_details = {"error": error_message, "last_state_history": history_summary}
        write_log("Max steps/consecutive failures reached. Returning error JSON.")
        return json.dumps(error_details, indent=4, ensure_ascii=False)
