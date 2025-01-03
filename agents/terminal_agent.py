import logging
import subprocess
import json
from typing import Any, Optional
from pydantic import BaseModel, ValidationError
from langchain_core.runnables import RunnableConfig
from agents.base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import asyncio

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CommandResult(BaseModel):
    command: str

class TerminalTool:
    def __init__(self):
        self.name = "terminal"

    def run(self, command: str) -> str:
        logging.info(f"Executing command: {command}")
        try:
            logging.info("###################################")
            logging.info(f"subprocess.run args: command={command}, shell=True, capture_output=True, text=True, check=True")
            logging.info("###################################")
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )
            logging.info(f"Command output: {process.stdout}")
            return process.stdout
        except subprocess.CalledProcessError as e:
            logging.error(f"Command error: {e.stderr}")
            return e.stderr

class TerminalAgent(BaseAgent):
    def __init__(self, llm=None, tools=None):
        super().__init__(llm, tools)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "あなたはターミナルコマンドを実行するエージェントです。ユーザーの指示に従って、適切なターミナルコマンドを生成し、以下のJSON形式で出力してください。\n\n```json\n{\"command\": \"生成するコマンド\"}\n```"),
            MessagesPlaceholder(variable_name="messages")
        ])

    def run(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        logging.info("TerminalAgent.run開始")
        # logging.debug(f"Input: {input}")

        if isinstance(input, str):
            logging.info(f"直接実行するコマンド: {input}")
            terminal_tool = next((t for t in (self.tools or []) if t.name == "terminal"), None)
            if terminal_tool:
                logging.debug("TerminalTool.run を呼び出します。")
                return terminal_tool.run(input)  # ここでコマンドが実行される
            else:
                logging.warning("TerminalToolが見つかりませんでした。コマンド実行スキップ。")
                return ""
        else:
            logging.info("###################################")
            logging.info(f"input: {input}")
            logging.info("###################################")
            logging.error("TerminalAgentは文字列のコマンド入力を期待しています。")
            return ""

    async def arun(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        logging.info("TerminalAgent.arun開始")
        return await asyncio.to_thread(self.run, input, config)