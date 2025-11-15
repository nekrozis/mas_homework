# src/mas_homework/tools/pdf_tools.py
import json
from pathlib import Path
from typing import Any, Dict

import fitz  # PyMuPDF

from ..tool_system import GLOBAL_TOOLS, tool
from ..types import ToolResult

# ==============================================================================
# PDF 文件工具定义 (修复 njoin 错误)
# ==============================================================================


@tool(
    GLOBAL_TOOLS,
    "pdf_read",
    "Read a specific page from a PDF file. Parameter 'page' is 1-indexed. Returns the first 10 lines as a snippet. E.g., path='name.pdf', page=1",
)
def pdf_read(path: str, page: int = 1) -> ToolResult:
    """读取 PDF 文件指定页面的内容，仅返回前十行作为摘要。"""
    pdf_path = Path(path)

    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return ToolResult(
            success=False,
            output=None,
            error=f"Invalid or non-existent PDF file: '{path}'",
        )

    try:
        doc = fitz.open(pdf_path.as_posix())

        doc_page_count = doc.page_count
        internal_page_index = page - 1

        if page < 1 or page > doc_page_count:
            doc.close()
            return ToolResult(
                success=False,
                output=None,
                error=f"Page {page} out of range. Document has {doc_page_count} pages.",
            )

        raw_output = doc.load_page(internal_page_index).get_text("text")
        text: str = str(raw_output)
        doc.close()

        snippet: str = "\n".join(text.splitlines()[:10])

        output: str = f"Content of page {page} in '{path}' (Total {len(text.splitlines())} lines. Snippet):\n{snippet}\n..."
        return ToolResult(success=True, output=output, error=None)

    except Exception as e:
        return ToolResult(
            success=False,
            output=None,
            error=f"Error reading PDF: {type(e).__name__}: {e}",
        )


@tool(
    GLOBAL_TOOLS,
    "pdf_text",
    "Extract all text content from a PDF file. Use this ONLY if partial reading is insufficient, as output can be very large. E.g., path='report.pdf'",
)
def pdf_text(path: str) -> ToolResult:
    """提取整个 PDF 文件的所有文本内容。"""
    pdf_path = Path(path)

    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return ToolResult(
            success=False,
            output=None,
            error=f"Invalid or non-existent PDF file: '{path}'",
        )

    full_text = []
    try:
        doc = fitz.open(pdf_path.as_posix())
        for i in range(doc.page_count):
            page_text = doc.load_page(i).get_text("text")
            full_text.append(page_text)

        doc.close()

        # 🚀 修复点: 将 njoin 改为 join
        output = "\n".join(full_text)
        # ---------------------------

        # 限制返回长度，避免输出过大导致 LLM 记忆溢出
        MAX_TEXT_LEN = 10000
        if len(output) > MAX_TEXT_LEN:
            # 使用 doc.page_count 之前确保 doc 没有被意外关闭或异常终止
            page_count_display = doc.page_count if "doc" in locals() else "N/A"
            output = (
                output[:MAX_TEXT_LEN]
                + f"\n\n[TRUNCATED] Text content was truncated to {MAX_TEXT_LEN} characters. Total pages: {page_count_display}."
            )

        return ToolResult(success=True, output=output, error=None)

    except Exception as e:
        return ToolResult(
            success=False,
            output=None,
            error=f"Error extracting full text from PDF: {type(e).__name__}: {e}",
        )


@tool(
    GLOBAL_TOOLS,
    "pdf_meta",
    "Extract structured metadata (Title, Author, Creation Date, Page Count, and first page snippet) from a PDF file. E.g., path='test.pdf'",
)
def pdf_meta(path: str) -> ToolResult:
    """从 PDF 文件中提取元数据。返回一个结构化的 JSON 字符串。"""
    pdf_path = Path(path)

    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return ToolResult(
            success=False,
            output=None,
            error=f"Invalid or non-existent PDF file: '{path}'",
        )

    try:
        doc = fitz.open(pdf_path.as_posix())

        # 1. 提取文档元数据
        metadata: Dict[str, Any] = doc.metadata or {}
        page_count: int = doc.page_count

        # 2. 提取摘要（尝试从第一页提取前几段作为摘要）
        raw_output = doc.load_page(0).get_text("text")
        first_page_text: str = str(raw_output)

        # 提取前 20 行作为摘要片段
        abstract_snippet: str = "\n".join(first_page_text.splitlines()[:20])

        doc.close()

        # 3. 构造结构化输出
        meta_data_output: Dict[str, Any] = {
            "file": path,
            "title": metadata.get("title", "N/A"),
            "author": metadata.get("author", "N/A"),
            "creation_date": metadata.get("creationDate", "N/A"),
            "page_count": page_count,
            "first_page_snippet": abstract_snippet,
        }

        output_json: str = json.dumps(meta_data_output, ensure_ascii=False, indent=4)
        return ToolResult(success=True, output=output_json, error=None)

    except Exception as e:
        return ToolResult(
            success=False,
            output=None,
            error=f"Error extracting metadata from '{path}': {type(e).__name__}: {e}",
        )
