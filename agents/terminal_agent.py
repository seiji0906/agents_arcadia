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
        # ChatPromptTemplateを使用してプロンプト作成
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "あなたは、ターミナルコマンドを生成するエージェントです。\n"
                "入力された要求に基づいて、実行可能なコマンドを生成してください。\n"
                "必ず以下のJSON形式で出力してください。他の形式は許可されません：\n\n"
                "```json\n{\"command\":\"生成されたコマンド\"}\n```\n\n"
                "例1: pythonスクリプトを実行する場合\n"
                "```json\n{\"command\":\"python script.py\"}\n```\n\n"
                "例2: ファイルを表示する場合\n"
                "```json\n{\"command\":\"cat file.txt\"}\n```\n\n"
                "注意：\n"
                "- 必ずJSONオブジェクトを返してください\n"
                "- 説明文や追加のテキストは含めないでください\n"
                "- commandキーのみを含むオブジェクトを返してください"
            ),
            MessagesPlaceholder(variable_name="messages")
        ])

    def run(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        logging.info("TerminalAgent.run開始")
        # logging.info(f"Input: {input}")

        # input は list 形式を想定
        if not isinstance(input, list):
            logging.warning("input は通常 list[BaseMessage] を想定しているため、型を確認してください。")
            return ""

        # 入力メッセージの詳細をログ出力
        logging.debug("=== TerminalAgent入力メッセージの詳細 ===")
        logging.debug(f"入力の型: {type(input)}")  # 入力の型をログ出力
        logging.debug(f"入力の内容: {input}")    # 入力の内容をログ出力
        if isinstance(input, list):
            for i, msg in enumerate(input):
                logging.debug(f"  [{i}] 要素の型: {type(msg)}")  # 各要素の型をログ出力
                if not isinstance(msg, BaseMessage):
                    logging.error(f"  [{i}] 警告: 要素は BaseMessage のインスタンスではありません。")
                logging.debug(f"  [{i}] Message Content: {msg.content}")
        logging.debug("================================")

        # プロンプトを整形してLLMに投入
        try:
            logging.debug("========hogehogehogeho==========")
            logging.debug(f"input: {input}")
            messages = self.prompt.format_messages(messages=input)
            logging.debug("=== LLMへ送信するプロンプト ===")
            for msg in messages:
                logging.debug(f"Message Type: {type(msg)}")
                logging.debug(f"Message Content: {msg.content}")
            logging.debug("================================")

            # LLM 呼び出し直前に messages と config の内容をログ出力
            logging.debug(f"LLMに送信する messages: {messages}")
            logging.debug(f"LLMに送信する config: {config}")

            response = self.llm.invoke(messages, config)
            logging.debug("========fugafugafugafuga==========")
            logging.debug(f"=== LLMからの生の応答 ===\n{response}\n====================")
            
            # レスポンスの内容をクリーンアップ
            content = response.content.strip()
            # コードブロックを除去
            if "```json" in content:
                content = content.split("```json")[-1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[-2].strip()
                
            logging.info(f"クリーンアップ後の内容: {content}")
            
            try:
                # 直接JSONとしてパースを試みる
                data = json.loads(content)
            except json.JSONDecodeError as e:
                logging.error(f"JSONのパース中にエラーが発生しました: {e}")
                logging.error(f"パースに失敗した文字列: {content}")
                return ""

            try:
                command_result = CommandResult(**data)
                generated_command = command_result.command
                logging.info(f"抽出されたコマンド: {generated_command}")
            except ValidationError as e:
                logging.error(f"CommandResultのバリデーションエラー: {e}")
                return ""

            # ターミナルツールで実行
            if generated_command:
                terminal_tool = next((t for t in (self.tools or []) if t.name == "terminal"), None)
                if terminal_tool:
                    return terminal_tool.run(generated_command)
                else:
                    logging.warning("TerminalToolが見つかりませんでした。コマンド実行スキップ。")
                    return ""
            else:
                logging.warning("生成されたコマンドが空です。実行をスキップします。")
                return ""
                
        except Exception as e:
            logging.error(f"予期せぬエラーが発生しました: {e}")
            return ""

    async def arun(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        # 非同期バージョンは同期処理を呼び出すか、実装を分ける
        logging.info("TerminalAgent.arun開始")
        return self.run(input, config) 