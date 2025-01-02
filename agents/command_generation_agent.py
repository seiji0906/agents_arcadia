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
            ("system", "あなたは有能なコマンド生成アシスタントです。ユーザーの指示に従って、実行可能なターミナルコマンドを生成し、以下の**厳密なJSON形式**で提供してください。\n\n```json\n{{\"command\": \"生成するコマンド\"}}\n```\n\n**JSONオブジェクトのみを返し、それ以外のテキストは含めないでください。**"),
            MessagesPlaceholder(variable_name="messages")
        ])

    def run(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        command_gen_messages = self.command_generation_prompt.format_messages(messages=input)
        logging.info(f"CommandGenerationAgent - command_gen_messages: {command_gen_messages}")
        response = self.llm.invoke(command_gen_messages, config)
        logging.info("##########################")
        logging.info(f"CommandGenerationAgent - LLM response: {response.content}")

        if not isinstance(response.content, str):
            raise ValueError("Unexpected response type (not a string).")

        # LLMの応答からコマンドを抽出
        content = response.content.strip()

        # JSON形式の開始と終了のインデックスを見つける
        start_index = content.find('{')
        end_index = content.rfind('}') + 1

        if start_index != -1 and end_index > start_index:
            content = content[start_index:end_index]
        else:
            logging.error(f"CommandGenerationAgent - JSON形式の文字列が見つかりませんでした: {response.content}")
            return json.dumps({"command": ""})

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
        logging.info(f"CommandGenerationAgent - command_gen_messages: {command_gen_messages}")
        response = await self.llm.ainvoke(command_gen_messages, config)

        logging.info(f"CommandGenerationAgent - LLM response: {response.content}")

        if not isinstance(response.content, str):
            raise ValueError("Unexpected response type (not a string).")

        # LLMの応答からコマンドを抽出
        content = response.content.strip()
        # JSON形式の開始と終了のインデックスを見つける
        start_index = content.find('{')
        end_index = content.rfind('}') + 1

        if start_index != -1 and end_index > start_index:
            content = content[start_index:end_index]
        else:
            logging.error(f"CommandGenerationAgent - JSON形式の文字列が見つかりませんでした (非同期): {response.content}")
            return json.dumps({"command": ""})

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