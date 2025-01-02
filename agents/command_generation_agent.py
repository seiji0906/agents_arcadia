# /agents/command_generation_agent.py
"""
CommandGenerationAgent: コマンド生成に特化したエージェント
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from agents.base_agent import BaseAgent
from langchain_core.runnables import RunnableConfig
from typing import Any, Optional, Literal
import json

class CommandGenerationAgent(BaseAgent):
    """
    コマンド生成に特化した機能を実装するエージェント。
    """

    def __init__(self, llm, tools=None):
        super().__init__(llm, tools)
        self.command_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", "あなたは有能なコマンド生成アシスタントです。ユーザーの指示に従って、実行可能なターミナルコマンドを生成し、以下の形式で提供してください。\n\n```json\n{{\"command\": \"生成するコマンド\"}}\n```"),
            MessagesPlaceholder(variable_name="messages")
        ])

    def run(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        command_gen_messages = self.command_generation_prompt.format_messages(messages=input)
        response = self.llm.invoke(command_gen_messages, config)

        if not isinstance(response.content, str):
            raise ValueError("Unexpected response type (not a string).")

        # 生成したコマンドをJSON形式で返す
        return json.dumps({"command": response.content})

    async def arun(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        command_gen_messages = self.command_generation_prompt.format_messages(messages=input)
        response = await self.llm.ainvoke(command_gen_messages, config)

        if not isinstance(response.content, str):
            raise ValueError("Unexpected response type (not a string).")

        # 生成したコマンドをJSON形式で返す
        return json.dumps({"command": response.content})