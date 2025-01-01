"""
BaseAgent: 抽象基底クラス
全てのエージェントに共通するロジック・インターフェースを定義します。
"""

from langchain_core.runnables import RunnableConfig
from typing import Any, Optional

class BaseAgent:
    """
    すべてのエージェントに共通するロジック・インターフェースを持つ抽象基底クラス。
    LLM呼び出し、ログ管理、ツール管理などの共通機能を集約します。
    """

    def __init__(self, llm, tools=None):
        self.llm = llm
        self.tools = tools or []
        self.tool_map = {tool.name: tool for tool in self.tools}

    def get_llm(self):
        """LLMインスタンスを取得する"""
        return self.llm

    def get_tools(self):
        """利用可能なツールのリストを取得する"""
        return self.tools

    def get_tool(self, tool_name: str):
        """指定された名前のツールを取得する"""
        return self.tool_map.get(tool_name)

    def run(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """エージェントを実行する"""
        raise NotImplementedError

    async def arun(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """エージェントを非同期で実行する"""
        raise NotImplementedError 