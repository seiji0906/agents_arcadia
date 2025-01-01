"""
ReviewAgent: 要件定義のレビューに特化したエージェント
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from agents.base_agent import BaseAgent
from langchain_core.runnables import RunnableConfig
from typing import Any, Optional

class ReviewAgent(BaseAgent):
    """
    要件定義のレビューに特化した機能を実装するエージェント。
    """

    def __init__(self, llm, tools=None):
        super().__init__(llm, tools)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful review assistant. Please review the following requirement definition and provide feedback."),
            MessagesPlaceholder(variable_name="messages")
        ])

    def run(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        messages = self.prompt.format_messages(messages=input)
        response = self.llm.invoke(messages, config)
        return response

    async def arun(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        messages = self.prompt.format_messages(messages=input)
        response = await self.llm.ainvoke(messages, config)
        return response 