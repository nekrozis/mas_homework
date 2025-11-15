# src/mas_homework/tools/pdf_tools.py
import json
import re
from pathlib import Path
from typing import Any, Dict, List, cast

import fitz  # PyMuPDF

from ..tool_system import GLOBAL_TOOLS, tool
from ..types import ToolResult


def _parse_pages_input(pages_input: str, max_page: int) -> List[int]:
    """
    解析页码输入字符串为合法的 0-indexed 页码列表。
    支持单页（"1"）、列表（"1,3,5"）和范围（"2-5"）。
    """
    if isinstance(pages_input, int):
        pages_input = str(pages_input)

    pages_input = pages_input.strip()
    valid_pages = set()
    parts = re.split(r",\s*", pages_input)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if re.match(r"^\d+$", part):
            p = int(part)
            if 1 <= p <= max_page:
                valid_pages.add(p)
        elif re.match(r"^\d+-\d+$", part):
            try:
                start, end = map(int, part.split("-"))
                if start > end:
                    start, end = end, start
                for p in range(start, end + 1):
                    if 1 <= p <= max_page:
                        valid_pages.add(p)
            except ValueError:
                continue

    return sorted([p - 1 for p in valid_pages])


@tool(
    GLOBAL_TOOLS,
    "pdf_pages",
    """Extract text content from specific pages of a PDF.

Parameters:
- path (str): Path to the PDF file.
- pages (str): Pages to extract, supports:
    1. Single page, e.g., "3"
    2. Comma-separated list, e.g., "1,3,5"
    3. Page range, e.g., "2-5"
    4. Combination, e.g., "1,3-4,6"

Returns:
- ToolResult:
    success (bool): Whether the extraction succeeded.
    output (str): Extracted text from the specified pages, with each page prefixed by "--- Page X/Y ---".
    error (str | None): Error message if failed.

Examples:
1. Extract a single page:
    pdf_pages(path="report.pdf", pages="2")

2. Extract multiple non-contiguous pages:
    pdf_pages(path="report.pdf", pages="1,3,5")

3. Extract a range of pages:
    pdf_pages(path="report.pdf", pages="2-4")

4. Extract a combination:
    pdf_pages(path="report.pdf", pages="1,3-5,7")
""",
)
def pdf_pages(path: str, pages: str) -> ToolResult:
    pdf_path = Path(path)
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return ToolResult(
            success=False, error=f"Invalid or non-existent PDF file: '{path}'"
        )

    try:
        doc = fitz.open(pdf_path.as_posix())
        max_page = doc.page_count
        page_indices = _parse_pages_input(pages, max_page)

        if not page_indices:
            doc.close()
            return ToolResult(
                success=False,
                error=f"No valid pages found in input '{pages}'. Document has {max_page} pages.",
            )

        extracted_texts = []
        for index in page_indices:
            text = cast(str, doc.load_page(index).get_text("text"))
            page_num = index + 1
            extracted_texts.append(f"--- Page {page_num}/{max_page} ---\n{text}")

        doc.close()

        output = "\n\n".join(extracted_texts)
        MAX_TEXT_LEN = 10000
        if len(output) > MAX_TEXT_LEN:
            output = (
                output[:MAX_TEXT_LEN]
                + f"\n\n[TRUNCATED] Text truncated to {MAX_TEXT_LEN} characters. Total pages extracted: {len(page_indices)}."
            )

        return ToolResult(success=True, output=output)

    except Exception as e:
        doc_obj = locals().get("doc")
        if doc_obj is not None and not doc_obj.is_closed:
            doc_obj.close()
        return ToolResult(
            success=False, error=f"Error reading PDF: {type(e).__name__}: {e}"
        )


@tool(
    GLOBAL_TOOLS,
    "pdf_meta",
    "Extract structured metadata (Title, Author, Creation Date, Page Count, and first page snippet) from a PDF file.",
)
def pdf_meta(path: str) -> ToolResult:
    pdf_path = Path(path)
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return ToolResult(
            success=False, error=f"Invalid or non-existent PDF file: '{path}'"
        )

    try:
        doc = fitz.open(pdf_path.as_posix())
        metadata = doc.metadata or {}
        page_count = doc.page_count
        first_page_text = cast(str, doc.load_page(0).get_text("text"))
        doc.close()

        abstract_snippet = "\n".join(first_page_text.splitlines()[:20])
        meta_data_output: Dict[str, Any] = {
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
        doc_obj = locals().get("doc")
        if doc_obj is not None and not doc_obj.is_closed:
            doc_obj.close()
        return ToolResult(
            success=False,
            error=f"Error extracting metadata from '{path}': {type(e).__name__}: {e}",
        )
