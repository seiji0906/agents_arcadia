import logging
from langchain_openai import ChatOpenAI
from agents.base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
import subprocess
from typing import Any, Optional
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ValidationError
import json

# ログの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("terminal_test.log"),
        logging.StreamHandler()
    ]
)

load_dotenv()

class CommandResult(BaseModel):
    command: str

# ターミナルコマンド実行ツール
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

# ターミナルコマンド実行エージェント
class TerminalAgent(BaseAgent):
    def __init__(self, llm, tools=None):
        super().__init__(llm, tools)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "あなたはターミナルコマンドを実行するエージェントです。ユーザーの指示に従って、適切なターミナルコマンドを生成し、以下のJSON形式で出力してください。\n\n```json\n{{\"command\": \"生成するコマンド\"}}\n```"),
            MessagesPlaceholder(variable_name="messages")
        ])

    def run(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        logging.info("TerminalAgent.run開始")
        logging.debug(f"Input: {input}")

        try:
            logging.debug("フォーマット前のメッセージ内容:")
            for msg in input:
                logging.debug(f"Message type: {type(msg)}, content: {msg.content}")

            messages = self.prompt.format_messages(messages=input)
            logging.debug(f"Formatted messages: {messages}")
        except KeyError as e:
            logging.error(f"メッセージのフォーマット中にキーエラーが発生しました: {e}")
            return ""

        try:
            response = self.llm.invoke(messages, config)
            logging.info(f"LLMからのレスポンス: {response.content}")
        except Exception as e:
            logging.error(f"LLMの呼び出し中にエラーが発生しました: {e}")
            return ""

        content = response.content.strip()
        logging.debug(f"レスポンス内容（トリム後）: {content}")

        # JSON部分のみを抽出
        start_index = content.find('{')
        end_index = content.rfind('}') + 1

        if start_index != -1 and end_index > start_index:
            json_string = content[start_index:end_index]
            logging.debug(f"抽出されたJSON文字列: {json_string}")
        else:
            logging.error("JSON形式の文字列が見つかりませんでした。")
            return ""

        try:
            data = json.loads(json_string)
            logging.debug(f"パースされたデータ: {data}")
            command_result = CommandResult(**data)
            logging.info(f"抽出されたコマンド: {command_result.command}")
            return command_result.command
        except (json.JSONDecodeError, ValidationError) as e:
            logging.error(f"JSONのパース中にエラーが発生しました: {e}")
            logging.error(f"パース失敗した文字列: {json_string}")
            return ""

    async def arun(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        logging.info("TerminalAgent.arun開始")
        logging.debug(f"Input: {input}")

        try:
            logging.debug("フォーマット前のメッセージ内容 (非同期):")
            for msg in input:
                logging.debug(f"Message type: {type(msg)}, content: {msg.content}")

            messages = self.prompt.format_messages(messages=input)
            logging.debug(f"Formatted messages (async): {messages}")
        except KeyError as e:
            logging.error(f"メッセージのフォーマット中にキーエラーが発生しました (非同期): {e}")
            return ""

        try:
            response = await self.llm.ainvoke(messages, config)
            logging.info(f"LLMからのレスポンス (非同期): {response.content}")
        except Exception as e:
            logging.error(f"LLMの非同期呼び出し中にエラーが発生しました: {e}")
            return ""

        content = response.content.strip()
        logging.debug(f"レスポンス内容（トリム後, 非同期）: {content}")

        # JSON部分のみを抽出
        start_index = content.find('{')
        end_index = content.rfind('}') + 1

        if start_index != -1 and end_index > start_index:
            json_string = content[start_index:end_index]
            logging.debug(f"抽出されたJSON文字列 (非同期): {json_string}")
        else:
            logging.error("JSON形式の文字列が見つかりませんでした。 (非同期)")
            return ""

        try:
            data = json.loads(json_string)
            logging.debug(f"パースされたデータ (非同期): {data}")
            command_result = CommandResult(**data)
            logging.info(f"抽出されたコマンド (非同期): {command_result.command}")
            return command_result.command
        except (json.JSONDecodeError, ValidationError) as e:
            logging.error(f"JSONのパース中にエラーが発生しました (非同期): {e}")
            logging.error(f"パース失敗した文字列: {json_string}")
            return ""

async def main():
    logging.info("main関数開始")
    # LLMの準備
    try:
        llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o",
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )
        logging.info("LLMの初期化に成功しました。")
    except Exception as e:
        logging.error(f"LLMの初期化中にエラーが発生しました: {e}")
        return

    # ツールの準備
    terminal_tool = TerminalTool()
    tools = [terminal_tool]
    logging.info("ツールの準備が完了しました。")

    # エージェントの準備
    terminal_agent = TerminalAgent(llm, tools)
    logging.info("TerminalAgentの準備が完了しました。")

    # 実行するコマンドの指示
    command_instruction = "fast_test.pyを実行してサーバーを立ち上げてください。"
    logging.info(f"コマンド指示: {command_instruction}")

    # エージェントにコマンドを生成させる
    try:
        generated_command = terminal_agent.run([HumanMessage(content=command_instruction)])
        logging.info(f"生成されたコマンド: {generated_command}")
    except Exception as e:
        logging.error(f"コマンド生成中にエラーが発生しました: {e}")
        generated_command = ""

    # ターミナルコマンドを実行
    if generated_command:
        try:
            result = terminal_tool.run(generated_command)
            logging.info(f"コマンド実行結果:\n{result}")
        except Exception as e:
            logging.error(f"コマンド実行中にエラーが発生しました: {e}")
    else:
        logging.warning("生成されたコマンドが空です。実行をスキップします。")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
