from agents.base_agent import BaseAgent
from langchain_core.runnables import RunnableConfig
from typing import Any, Optional
import json
import os

class FileOperationAgent(BaseAgent):
    """
    コーディングエージェントの出力（JSON形式）を渡すと、
    実際にファイルの読み書きを行うかどうかを決定し、必要に応じて実行するエージェント。
    """

    def run(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        raw_text = str(input)
        return self._process_file_operation(raw_text)

    async def arun(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        raw_text = str(input)
        return self._process_file_operation(raw_text)

    def _process_file_operation(self, raw_text: str) -> str:
        # ファイルパスとコード内容をテキストから抽出
        file_path = None
        code_content = None

        lines = raw_text.splitlines()
        for line in lines:
            if line.startswith("ファイルパス:"):
                file_path = line.split(":", 1)[1].strip()
            elif line.startswith("コード:"):
                code_content = line.split(":", 1)[1].strip()

        if not file_path or not code_content:
            return "出力テキストからファイルパスまたはコードの内容を抽出できませんでした。"

        action = "modify" # デフォルトでmodifyとする

        # 実際にファイルへ書き込むなどの処理
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code_content + "\n")
        except OSError as e:
            return f"ファイル書き込みに失敗しました: {e}"

        return f"ファイル '{file_path}' にコードを書き込みました。" 