"""
CodingAgent: コード生成・修正に特化したエージェント
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from agents.base_agent import BaseAgent
from langchain_core.runnables import RunnableConfig
from typing import Any, Optional, Literal
import logging
class CodingAgent(BaseAgent):
    """
    コード生成・修正に特化した機能を実装するエージェント。
    """

    def __init__(self, llm, tools=None):
        super().__init__(llm, tools)
        self.coding_prompt = ChatPromptTemplate.from_messages([
            ("system", "あなたは有能なコーディングアシスタントです。ユーザーの指示に従ってコードを生成し、ファイルパスとコードを以下の形式で提供してください。\n\nファイルパス: <生成するファイルのパス>\nコード: \n```\n<生成されたコード>\n```"),
            MessagesPlaceholder(variable_name="messages")
        ])

    def run(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        logging.info("########################## coding_agent")
        logging.info(f"CodingAgent - input: {input}")
        code_gen_messages = self.coding_prompt.format_messages(messages=input)
        response = self.llm.invoke(code_gen_messages, config)

        if not isinstance(response.content, str):
            raise ValueError("Unexpected response type (not a string).")

        # 生成したテキストをそのまま返すのみ
        return response.content

    async def arun(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        code_gen_messages = self.coding_prompt.format_messages(messages=input)
        response = await self.llm.ainvoke(code_gen_messages, config)

        if not isinstance(response.content, str):
            raise ValueError("Unexpected response type (not a string).")

        # 生成したテキストをそのまま返すのみ
        return response.content 