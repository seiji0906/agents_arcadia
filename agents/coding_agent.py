"""
CodingAgent: コード生成・修正に特化したエージェント
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from agents.base_agent import BaseAgent
from langchain_core.runnables import RunnableConfig
from typing import Any, Optional, Literal

class CodingAgent(BaseAgent):
    """
    コード生成・修正に特化した機能を実装するエージェント。
    coding_promptのみで、'file_path', 'code', 'action', 'description' の4つのキーが
    必ず含まれるJSONを返すように指示します。
    """

    def __init__(self, llm, tools=None):
        super().__init__(llm, tools)
        self.coding_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a coding assistant that MUST respond in JSON format.
The output must have these keys:
- file_path
- code
- action (must be 'create' or 'modify')
- description

Additional requirements:
1. Response MUST be a single, valid JSON object
2. No text before or after the JSON
3. No markdown, code blocks, or backticks
4. Use double quotes for all strings
5. Properly escape all special characters
6. No comments or trailing commas
7. The 'action' field must be exactly 'create' or 'modify'"""
            ),
            MessagesPlaceholder(variable_name="messages")
        ])

    def run(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        code_gen_messages = self.coding_prompt.format_messages(messages=input)
        response = self.llm.invoke(code_gen_messages, config)

        if not isinstance(response.content, str):
            raise ValueError("Unexpected response type (not a string).")

        # 生成したJSONらしきテキストをそのまま返すのみ
        return response.content

    async def arun(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        code_gen_messages = self.coding_prompt.format_messages(messages=input)
        response = await self.llm.ainvoke(code_gen_messages, config)

        if not isinstance(response.content, str):
            raise ValueError("Unexpected response type (not a string).")

        # 生成したJSONらしきテキストをそのまま返すのみ
        return response.content 