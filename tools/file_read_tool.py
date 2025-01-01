"""
FileReadTool: 指定ファイルの内容を読み込むツール
"""

from langchain_core.tools import BaseTool
from typing import Any

class FileReadTool(BaseTool):
    """Read text content from a specified file path."""
    name: str = "file_read"
    description: str = "Read text content from a specified file path."

    def _run(self, file_path: str) -> str:
        """同期処理でファイルを読み込む"""
        import os
        if not os.path.exists(file_path):
            print(f"警告: ファイル {file_path} が存在しません。")
            return ""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    async def _arun(self, file_path: str) -> str:
        """非同期処理でファイルを読み込む"""
        import os
        if not os.path.exists(file_path):
            print(f"警告: ファイル {file_path} が存在しません。")
            return ""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read() 