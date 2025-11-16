# mas-homework — 本地 ReAct 文档处理 Agent

本项目实现了一个基于 ReAct（Reasoning + Acting）流程的本地文档分析 Agent。核心逻辑使用本机的 Ollama 模型进行推理，当模型判断需要使用工具时，会自动发起调用，并在执行前进行人工确认。项目使用 `uv` 管理环境与运行。

## 功能概述

- 使用 ReAct 工作流执行“推理 → 工具 → 推理”循环。
- LLM 输出必须是 JSON，包含 `thought` 字段，并在 `tool_call` 与 `final_answer` 中任选其一。
- 内置文件系统浏览工具与 PDF 提取工具（`ls`、`pdf_meta`、`pdf_pages`）。
- 工具调用需人工确认（Human-in-the-loop）。
- 对 LLM 输出进行 Pydantic 校验，异常会被记录、反馈，并在连续失败达到阈值时中止循环。
- 所有执行步骤写入日志与历史文件，可用于调试或回溯。
- 全程本地运行，不需要联网。

---

## 项目结构

```
.
├── data/                     # 示例文件目录
├── pyproject.toml
├── README.md
├── src/
│   ├── main.py               # 命令行入口 & 交互模式入口
│   └── mas_homework/
│       ├── agent_core.py     # 核心逻辑：ReAct 主循环、日志、历史、校验等
│       ├── llm_client.py     # 本地 Ollama 调用封装
│       ├── tool_system.py    # 工具注册、查找与调度
│       ├── tools/
│       │   ├── fs_tools.py   # 文件系统工具：ls
│       │   └── pdf_tools.py  # PDF 工具：pdf_meta / pdf_pages
│       └── types.py          # Pydantic 模型与数据结构
└── uv.lock
```

---

## 环境准备

1. 安装并启动 Ollama，并确保所需模型已在本地拉取。
2. 安装 `uv`。
3. 在项目根目录执行：

```
uv sync
```

---

## 运行方式

### 一次性执行

```
uv run python src/main.py "列出 data 下的 PDF 并展示第一页内容"
```

指定模型（可选）：

```
uv run python src/main.py --model qwen3:1.7b "读取某个 PDF 的元数据"
```

### 交互模式

```
uv run python src/main.py
```

未传入任务字符串时，将进入循环交互模式。

---

## 已提供的工具

### 文件系统（fs_tools）

#### `ls(path_or_pattern: str = ".")`
列出目录内容或按通配符过滤文件，例如：

```
{"name": "ls", "arguments": {"path_or_pattern": "data/*.pdf"}}
```

---

### PDF 工具（pdf_tools）

#### `pdf_meta(path: str)`
读取 PDF 基础信息（页数、元数据）与首页内容片段。

#### `pdf_pages(path: str, pages: str)`
提取指定页文本（支持 `"1"`、`"1-3"`、`"1,3,5"`）。

---

## LLM 输出格式（ReAct JSON）

每一步输出必须是一个 JSON 对象，包含：

- `thought`: 字符串，模型用于解释当前思考过程  
- 二选一：
  - `tool_call`: 调用工具  
  - `final_answer`: 输出最终任务结果  

示例：工具调用

```
{
  "thought": "需要查看 PDF 列表以决定下一步。",
  "tool_call": {
    "name": "ls",
    "arguments": { "path_or_pattern": "data/*.pdf" }
  }
}
```

示例：最终回答

```
{
  "thought": "信息已整理完成。",
  "final_answer": "{\"summary\": \"...\"}"
}
```

---

## 实现细节

Agent 对 LLM 输出进行严格的 Pydantic 校验。  
若 JSON 解析或结构校验失败：

- 会在日志中记录原始内容
- 会将错误作为 Observation 返回给模型
- 连续错误超过 `max_consecutive_failures` 后触发熔断，终止 ReAct 循环

---

## 人工确认（Human-in-the-loop）

调用任何工具前，控制台会出现提示：

```
[CONFIRM] Execute tool 'ls' with args {'path_or_pattern': 'data/*.pdf'}?
Confirm? (y/n):
```

- 输入 `y` 执行工具  
- 输入其他值（或回车）将取消此次工具调用，并记录为 FAILED Observation

---

## 日志、历史文件（重要）

运行过程中会生成两个文件，**都位于当前工作目录（CWD）**：

```
agent.log           # 追加日志，包括时间、LLM 输出、工具调用等
agent_history.json  # 完整的 ReAct 步骤记录（thought、tool_call、observation）
```

例如：  
如果你在其他目录运行：

```
cd /tmp
uv run python ~/mas_homework/src/main.py "..."
```

日志与历史文件会出现在 `/tmp/` 下，而不是项目目录。

**清理方式：**

- 删除 `agent_history.json` 可清空历史，重新开始干净的会话。
- `agent.log` 可按需删除或归档。

---

## 常见问题 / 使用建议

- **路径问题**  
  所有相对路径均以“当前工作目录”为基准，而不是项目根目录。  
  若工具提示文件不存在，请检查是否需要使用绝对路径。

- **非规范 JSON**  
  若 LLM 输出不满足字段要求，Agent 会反馈错误、记录日志，并可能因连续失败而终止。

- **循环检测**  
  Agent 会检测重复工具调用或来回循环；若发生，会在日志中记录 "LOOP DETECTED"。

- **调试建议**  
  检查：
  - `agent.log`（原始模型输出、异常信息）
  - `agent_history.json`（工具调用序列）

---

## 开发者提示：如何添加新工具

1. 在 `src/mas_homework/tools/` 下创建新的 `.py` 文件。
2. 使用装饰器注册：

   ```
   @tool(GLOBAL_TOOLS, "name", "description")
   ```
3. 工具函数返回：

   ```
   ToolResult(success: bool, output: str | None, error: str | None)
   ```
4. 工具会自动加入系统提示，无需修改 agent_core 或主循环。

---

## 快速命令（汇总）

```
uv sync

uv run python src/main.py "分析某个 PDF"

uv run python src/main.py --model qwen3:1.7b "列出 data/*.pdf"

uv run python src/main.py     # 进入交互模式
```

---

## 许可证

MIT
