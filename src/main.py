# main.py
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# ======================================================================
# 路径查找修复: 确保 mas_homework 包可被查找
# ======================================================================

# 获取当前文件（main.py）所在的目录
CURRENT_FILE_DIR = Path(__file__).parent.resolve()

# 将当前目录（包含 mas_homework 包的目录）添加到 sys.path
if str(CURRENT_FILE_DIR) not in sys.path:
    sys.path.append(str(CURRENT_FILE_DIR))

# ======================================================================
# 绝对导入，消除 Pylance 警告
# ======================================================================

from mas_homework.agent_core import Agent
from mas_homework.llm_client import OllamaClient
from mas_homework.tools import fs_tools, pdf_tools

DEFAULT_MODEL = "llama3"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Run a ReAct Agent for document analysis.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "query", nargs="?", default=None, help="Initial user query or task (optional)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"The Ollama model to use. (Default: {DEFAULT_MODEL})",
    )

    return parser.parse_args()


async def run_agent_session(agent: Agent, initial_query: str):
    """运行 Agent 的交互式会话循环"""

    # 1. 运行初始指令
    if initial_query:
        print("-" * 50)
        print(f"Starting Agent with Initial Query: {initial_query}")
        print("-" * 50)
        final_result_str = await agent.run(initial_query)

        print_result(final_result_str)
    else:
        print("\n--- No initial query provided. Entering interactive mode. ---")

    # 2. 交互式循环
    while True:
        try:
            # 等待用户输入下一个指令
            user_input = input("Agent >>> ")
            if not user_input.strip():
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting session.")
                break

            print("-" * 50)
            print(f"New Query: {user_input}")
            print("-" * 50)

            # 每次循环都重新运行 Agent（AgentState在内部重置）
            final_result_str = await agent.run(user_input)
            print_result(final_result_str)

        except EOFError:
            print("\nExiting session.")
            break
        except KeyboardInterrupt:
            print("\nAgent stopped by user (Ctrl+C). Exiting.")
            break
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during the loop: {e}")


def print_result(final_result_str: str):
    """格式化并打印最终结果"""
    print("\n" + "=" * 50)
    print("           ✨ TASK COMPLETED ✨")
    print("=" * 50)
    try:
        # 尝试美化输出 JSON
        final_result_json = json.loads(final_result_str)
        print(json.dumps(final_result_json, indent=4, ensure_ascii=False))
    except (json.JSONDecodeError, TypeError):
        # 如果不是有效的 JSON，直接输出原始字符串
        print("Final Result (Raw):")
        print(final_result_str)
    print("=" * 50)


async def main():
    args = parse_args()
    model_name = args.model
    initial_query = args.query

    # 打印模型使用说明
    print(f"\n--- Ollama Model Setting ---")
    print(f"Model selected: {model_name}")
    print(f"Note: The Agent will attempt to use this name directly via the Ollama API.")
    print(f"----------------------------")

    # 1. 初始化 LLM 客户端
    try:
        llm_client = OllamaClient()
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize OllamaClient. Is Ollama running?")
        print(f"Details: {e}")
        sys.exit(1)

    # 2. 初始化 Agent
    # 每次运行 Agent.run() 都会从新的 AgentState 开始，实现隔离
    agent = Agent(llm_client=llm_client, model=model_name, max_steps=15)

    # 3. 启动交互式会话
    await run_agent_session(agent, initial_query)


def run_main():
    """入口点函数，处理 asyncio"""
    try:
        # 使用 run_main 作为实际的启动入口
        asyncio.run(main())
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred in run_main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_main()
