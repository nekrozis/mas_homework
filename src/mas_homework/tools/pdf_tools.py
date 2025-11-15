# src/mas_homework/tools/pdf_tools.py
import json
from pathlib import Path
from typing import Any, Dict

import fitz  # PyMuPDF

from ..tool_system import GLOBAL_TOOLS, tool
from ..types import ToolResult


@tool(
    GLOBAL_TOOLS,
    "pdf_read",
    "Read a specific page from a PDF file. Parameter 'page' is 1-indexed. Returns the first 10 lines as a snippet.",
)
def pdf_read(path: str, page: int = 1) -> ToolResult:
    """读取 PDF 文件指定页面的内容，仅返回前十行作为摘要"""
    pdf_path = Path(path)
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return ToolResult(
            success=False, error=f"Invalid or non-existent PDF file: '{path}'"
        )

    try:
        doc = fitz.open(pdf_path.as_posix())
        if page < 1 or page > doc.page_count:
            doc.close()
            return ToolResult(
                success=False,
                error=f"Page {page} out of range. Document has {doc.page_count} pages.",
            )

        text = doc.load_page(page - 1).get_text("text")
        doc.close()
        snippet = "\n".join(text.splitlines()[:10])
        output = f"Content of page {page} in '{path}' (Total {len(text.splitlines())} lines):\n{snippet}\n..."
        return ToolResult(success=True, output=output)

    except Exception as e:
        return ToolResult(
            success=False, error=f"Error reading PDF: {type(e).__name__}: {e}"
        )


@tool(
    GLOBAL_TOOLS,
    "pdf_text",
    "Extract all text content from a PDF file. Output can be large. Use only if partial reading is insufficient.",
)
def pdf_text(path: str) -> ToolResult:
    """提取整个 PDF 文件的文本内容"""
    pdf_path = Path(path)
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return ToolResult(
            success=False, error=f"Invalid or non-existent PDF file: '{path}'"
        )

    try:
        doc = fitz.open(pdf_path.as_posix())
        full_text = [doc.load_page(i).get_text("text") for i in range(doc.page_count)]
        doc.close()

        output = "\n".join(full_text)
        MAX_TEXT_LEN = 10000
        if len(output) > MAX_TEXT_LEN:
            output = (
                output[:MAX_TEXT_LEN]
                + f"\n\n[TRUNCATED] Text truncated to {MAX_TEXT_LEN} characters."
            )
        return ToolResult(success=True, output=output)

    except Exception as e:
        return ToolResult(
            success=False,
            error=f"Error extracting full text from PDF: {type(e).__name__}: {e}",
        )


@tool(
    GLOBAL_TOOLS,
    "pdf_meta",
    "Extract structured metadata (Title, Author, Creation Date, Page Count, and first page snippet) from a PDF file.",
)
def pdf_meta(path: str) -> ToolResult:
    """提取 PDF 文件元数据，返回结构化 JSON"""
    pdf_path = Path(path)
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return ToolResult(
            success=False, error=f"Invalid or non-existent PDF file: '{path}'"
        )

    try:
        doc = fitz.open(pdf_path.as_posix())
        metadata = doc.metadata or {}
        page_count = doc.page_count
        first_page_text = doc.load_page(0).get_text("text")
        doc.close()

        abstract_snippet = "\n".join(first_page_text.splitlines()[:20])
        meta_data_output = {
            "file": path,
            "title": metadata.get("title", "N/A"),
            "author": metadata.get("author", "N/A"),
            "creation_date": metadata.get("creationDate", "N/A"),
            "page_count": page_count,
            "first_page_snippet": abstract_snippet,
        }
        return ToolResult(
            success=True,
            output=json.dumps(meta_data_output, ensure_ascii=False, indent=4),
        )

    except Exception as e:
        return ToolResult(
            success=False,
            error=f"Error extracting metadata from '{path}': {type(e).__name__}: {e}",
        )
