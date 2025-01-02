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
import logging

class CommandGenerationAgent(BaseAgent):
    """
    コマンド生成に特化した機能を実装するエージェント。
    """

    def __init__(self, llm, tools=None):
        super().__init__(llm, tools)
        self.command_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", "あなたは有能なコマンド生成アシスタントです。ユーザーの指示に従って、実行可能なターミナルコマンドを生成し、以下の形式で提供してください。\n\n```json\n{\"command\": \"生成するコマンド\"}\n```\n\n必ず'command'キーのみを含むJSONオブジェクトを返してください。説明や追加のテキストは含めないでください。JSONオブジェクト以外の形式で返信しないでください。"),
            MessagesPlaceholder(variable_name="messages")
        ])

    def run(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        command_gen_messages = self.command_generation_prompt.format_messages(messages=input)
        response = self.llm.invoke(command_gen_messages, config)

        logging.info(f"CommandGenerationAgent - LLM response: {response.content}")

        if not isinstance(response.content, str):
            raise ValueError("Unexpected response type (not a string).")

        # LLMの応答からコマンドを抽出
        content = response.content.strip()

        # 回答がコードブロックで囲まれているか確認する
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        elif content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()

        try:
            data = json.loads(content)
            if "command" in data:
                return json.dumps({"command": data["command"]})
            else:
                logging.error(f"CommandGenerationAgent - 'command' key not found in response: {content}")
                return json.dumps({"command": ""})
        except json.JSONDecodeError as e:
            logging.error(f"CommandGenerationAgent - JSON decoding error: {e}")
            logging.error(f"Failed to parse: {content}")
            return json.dumps({"command": ""})

    async def arun(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        command_gen_messages = self.command_generation_prompt.format_messages(messages=input)
        response = await self.llm.ainvoke(command_gen_messages, config)

        logging.info(f"CommandGenerationAgent - LLM response: {response.content}")

        if not isinstance(response.content, str):
            raise ValueError("Unexpected response type (not a string).")

        # LLMの応答からコマンドを抽出
        content = response.content.strip()
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        elif content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()

        try:
            data = json.loads(content)
            if "command" in data:
                return json.dumps({"command": data["command"]})
            else:
                logging.error(f"CommandGenerationAgent - 'command' key not found in response: {response.content}")
                return json.dumps({"command": ""})
        except json.JSONDecodeError as e:
            logging.error(f"CommandGenerationAgent - JSON decoding error: {e}")
            logging.error(f"Failed to parse: {response.content}")
            return json.dumps({"command": ""})