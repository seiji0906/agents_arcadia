"""
PlanningAgent: タスク理解と要件定義に特化したエージェント
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from agents.base_agent import BaseAgent
from langchain_core.runnables import RunnableConfig
from typing import Any, Optional

class PlanningAgent(BaseAgent):
    """
    タスクの理解と要件定義に特化した機能を実装するエージェント。
    """

    def __init__(self, llm, tools=None):
        super().__init__(llm, tools)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful planning assistant. Please analyze the task and create a requirement definition."),
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