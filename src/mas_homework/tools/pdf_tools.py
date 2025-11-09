# src/mas_homework/tools/pdf_tools.py
import json
import os
from typing import Any, Dict, Union

import fitz

from ..tool_system import GLOBAL_TOOLS, ToolResult, tool

# ==============================================================================
# PDF 文件工具定义
# ==============================================================================


@tool(
    GLOBAL_TOOLS,
    "pdfread",
    "Read a specific page from a PDF file. E.g., path='name.pdf', page=1",
)
def pdfread(path: str, page: int = 1) -> ToolResult:
    """读取 PDF 文件指定页面的内容，仅返回前十行作为摘要。"""
    if not os.path.exists(path) or not path.lower().endswith(".pdf"):
        return {
            "success": False,
            "output": None,
            "error": f"Invalid or non-existent PDF file: '{path}'",
        }

    try:
        doc = fitz.open(path)
        if page < 1 or page > doc.page_count:
            doc.close()
            return {
                "success": False,
                "output": None,
                "error": f"Page {page} out of range. Document has {doc.page_count} pages.",
            }

        # 使用 "text" 模式
        raw_output = doc.load_page(page - 1).get_text("text")
        text: str = str(raw_output)
        doc.close()

        # 确保 text 是字符串，splitlines 方法就可以安全访问
        snippet: str = "\n".join(text.splitlines()[:10])

        output: str = f"Content of page {page} in '{path}' (Snippet):\n{snippet}..."
        return {"success": True, "output": output, "error": None}
    except Exception as e:
        return {
            "success": False,
            "output": None,
            "error": f"Error reading PDF: {type(e).__name__}: {e}",
        }


@tool(
    GLOBAL_TOOLS,
    "pdf_meta",
    "Extract structured metadata (Title, Author, Abstract snippet) from a PDF file. E.g., path='test.pdf'",
)
def pdf_meta(path: str) -> ToolResult:
    """
    从 PDF 文件中提取真实的元数据（标题、作者、创建日期、第一页摘要）。
    """
    if not os.path.exists(path) or not path.lower().endswith(".pdf"):
        return {
            "success": False,
            "output": None,
            "error": f"Invalid or non-existent PDF file: '{path}'",
        }

    try:
        doc = fitz.open(path)

        # 1. 提取文档元数据
        metadata: Dict[str, Any] = doc.metadata or {}
        page_count: int = doc.page_count

        # 2. 提取摘要（尝试从第一页提取前几段作为摘要）
        raw_output = doc.load_page(0).get_text("text")
        first_page_text: str = str(raw_output)

        abstract_snippet: str = "\n".join(first_page_text.splitlines()[:20])

        doc.close()

        # 3. 构造结构化输出
        meta_data_output: Dict[str, Union[str, int]] = {
            "file": path,
            "title": metadata.get("title", "N/A"),
            "author": metadata.get("author", "N/A"),
            "creation_date": metadata.get("creationDate", "N/A"),
            "page_count": page_count,
            "first_page_snippet": abstract_snippet,
        }

        output_json: str = json.dumps(meta_data_output, ensure_ascii=False, indent=4)
        return {"success": True, "output": output_json, "error": None}

    except Exception as e:
        return {
            "success": False,
            "output": None,
            "error": f"Error extracting metadata from '{path}': {type(e).__name__}: {e}",
        }
