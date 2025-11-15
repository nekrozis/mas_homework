# main.py
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# 确保 mas_homework 包可被找到
CURRENT_FILE_DIR = Path(__file__).parent.resolve()
if str(CURRENT_FILE_DIR) not in sys.path:
    sys.path.append(str(CURRENT_FILE_DIR))

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
    if initial_query:
        print("-" * 50)
        print(f"Starting Agent with Initial Query: {initial_query}")
        print("-" * 50)
        final_result_str = await agent.run(initial_query)
        print_result(final_result_str)
    else:
        print("\n--- No initial query provided. Entering interactive mode. ---")

    while True:
        try:
            user_input = input("Agent >>> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting session.")
                break

            print("-" * 50)
            print(f"New Query: {user_input}")
            print("-" * 50)

            final_result_str = await agent.run(user_input)
            print_result(final_result_str)

        except (EOFError, KeyboardInterrupt):
            print("\nExiting session.")
            break
        except Exception as e:
            print(f"[ERROR] Unexpected error during loop: {e}")


def print_result(final_result_str: str):
    """格式化并打印最终结果"""
    print("\n" + "=" * 50)
    print("           ✨ TASK COMPLETED ✨")
    print("=" * 50)
    try:
        # 尝试以漂亮的 JSON 格式打印
        final_result_json = json.loads(final_result_str)
        print(json.dumps(final_result_json, indent=4, ensure_ascii=False))
    except (json.JSONDecodeError, TypeError):
        print("Final Result (Raw):")
        print(final_result_str)
    print("=" * 50)


async def main():
    args = parse_args()
    model_name = args.model
    initial_query = args.query

    print(f"\n--- Ollama Model Setting ---")
    print(f"Model selected: {model_name}")
    print(f"Note: The Agent will attempt to use this name via the Ollama API.")
    print(f"----------------------------")

    try:
        llm_client = OllamaClient()
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize OllamaClient. Is Ollama running?")
        print(f"Details: {e}")
        sys.exit(1)

    # 初始化 Agent，每次 run() 使用独立状态
    agent = Agent(llm_client=llm_client, model=model_name, max_steps=15)

    await run_agent_session(agent, initial_query)


def run_main():
    """入口点函数，处理 asyncio"""
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n[FATAL ERROR] Unexpected error in run_main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_main()
