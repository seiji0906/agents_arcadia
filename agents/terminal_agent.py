import logging
import subprocess
import json
from typing import Any, Optional
from pydantic import BaseModel, ValidationError
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from agents.base_agent import BaseAgent

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
            process = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            logging.info(f"Command output: {process.stdout}")
            return process.stdout
        except subprocess.CalledProcessError as e:
            logging.error(f"Command error: {e.stderr}")
            return e.stderr

class TerminalAgent(BaseAgent):
    def __init__(self, llm=None, tools=None):
        super().__init__(llm, tools)

    def run(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        logging.info("TerminalAgent.run開始")

        # inputが文字列の場合、直接コマンドとして扱う
        if isinstance(input, str):
            logging.info("###################################")
            logging.info(f"直接実行するコマンド: {input}")
            terminal_tool = next((t for t in (self.tools or []) if t.name == "terminal"), None)
            if terminal_tool:
                return terminal_tool.run(input)
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
        # 非同期バージョンは同期処理を呼び出すか、実装を分ける
        logging.info("TerminalAgent.arun開始")
        return self.run(input, config)