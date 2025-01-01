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
        # JSONとしてパースを試みる
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            return f"JSONのパースに失敗しました: {e}"

        # 必要なキーの存在確認などを行う
        action = data.get("action")
        file_path = data.get("file_path")
        code_content = data.get("code", "")
        if not file_path or not code_content:
            return "JSONの中にfile_pathやcodeが含まれていません。"

        if action not in ("create", "modify"):
            return f"actionフィールドの値が不正です: {action}"

        # 実際にファイルへ書き込むなどの処理
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code_content + "\n")
        except OSError as e:
            return f"ファイル書き込みに失敗しました: {e}"

        return f"ファイル '{file_path}' にコードを書き込みました。" 