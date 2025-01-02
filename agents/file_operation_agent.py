from agents.base_agent import BaseAgent
from langchain_core.runnables import RunnableConfig
from typing import Any, Optional
import json
from langchain_core.prompts import ChatPromptTemplate
import logging
import aiofiles

class FileOperationAgent(BaseAgent):
    """
    コーディングエージェントの出力（テキスト形式）を渡すと、
    ファイルパスとコード内容を抽出し、JSON形式で出力するエージェント。
    """

    def __init__(self, llm, tools=None):
        super().__init__(llm, tools)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "あなたはコーディングエージェントの出力を解析し、ファイルパスとコード内容を抽出するアシスタントです。"),
            ("user", "コーディングエージェントの出力: {raw_text}"),
            ("system", "抽出されたfile_pathとcodeをJSON形式で出力してください。")
        ])

    def run(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        import logging
        # logging.basicConfig(level=logging.INFO)
        # logging.info(f"FileOperationAgent run開始: input={input}")
        raw_text = str(input)
        messages = self.prompt.format_messages(raw_text=raw_text)
        response = self.llm.invoke(messages, config)
        # logging.info(f"LLM response content: {response.content}")

        try:
            # logging.info(f"JSONパース前: {response.content}")  # JSONパース前の内容を出力
            # JSON形式の開始と終了のインデックスを見つける
            start_index = response.content.find('{')
            end_index = response.content.rfind('}') + 1

            if start_index != -1 and end_index > start_index:
                json_string = response.content[start_index:end_index]
                # logging.info(f"抽出されたJSON文字列: {json_string}")
                json_output = json.loads(json_string)
            else:
                logging.error(f"JSON形式の文字列が見つかりませんでした: {response.content}")
                return "出力テキストからJSON形式の文字列を抽出できませんでした。"

            file_path = json_output.get("file_path")
            code_content = json_output.get("code")
            logging.info(f"抽出されたfile_path: {file_path}")
            # logging.info(f"抽出されたcode_content: {code_content}")

            if not file_path or not code_content:
                return "JSON出力からファイルパスまたはコードの内容を抽出できませんでした。"

            # ファイル書き込み処理
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(code_content)
                result = f"{file_path} にコードを書き込みました。"
                logging.info(f"FileOperationAgent run終了: result={result}")
                return result
            except Exception as e:
                logging.error(f"ファイルの書き込みエラー: {e}")
                return f"ファイルの書き込みに失敗しました: {e}"

        except json.JSONDecodeError as e:
            logging.error(f"JSONパースエラー: {e}")  # エラー内容を出力
            logging.error(f"パース失敗した文字列: {response.content}")  # パースに失敗した文字列を出力
            return "出力テキストがJSON形式ではありませんでした。"

    async def arun(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        import logging
        logging.basicConfig(level=logging.INFO)
        # logging.info(f"FileOperationAgent arun開始: input={input}")
        raw_text = str(input)
        messages = self.prompt.format_messages(raw_text=raw_text)
        response = await self.llm.ainvoke(messages, config)
        # logging.info(f"LLM response content: {response.content}")

        try:
            # logging.info(f"JSONパース前 (非同期): {response.content}")  # JSONパース前の内容を出力
            # JSON形式の開始と終了のインデックスを見つける
            start_index = response.content.find('{')
            end_index = response.content.rfind('}') + 1

            if start_index != -1 and end_index > start_index:
                json_string = response.content[start_index:end_index]
                # logging.info(f"抽出されたJSON文字列 (非同期): {json_string}")
                json_output = json.loads(json_string)
            else:
                logging.error(f"JSON形式の文字列が見つかりませんでした (非同期): {response.content}")
                return "出力テキストからJSON形式の文字列を抽出できませんでした。"

            file_path = json_output.get("file_path")
            code_content = json_output.get("code")

            if not file_path or not code_content:
                logging.info("FileOperationAgent: JSON出力からファイルパスまたはコードの内容を抽出できませんでした。")
                return "JSON出力からファイルパスまたはコードの内容を抽出できませんでした。"

            # ファイル書き込み処理 (非同期)
            try:
                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(code_content)
                result = f"{file_path} にコードを書き込みました (非同期)。"
                # logging.info(f"FileOperationAgent arun終了 (非同期): result={result}")
                return result
            except Exception as e:
                logging.error(f"ファイルの書き込みエラー (非同期): {e}")
                return f"ファイルの書き込みに失敗しました (非同期): {e}"

        except json.JSONDecodeError as e:
            logging.error(f"JSONパースエラー (非同期): {e}")  # エラー内容を出力
            logging.error(f"パース失敗した文字列 (非同期): {response.content}")  # パースに失敗した文字列を出力
            return "出力テキストがJSON形式ではありませんでした。"

    def _process_file_operation(self, raw_text: str) -> str:
        # このメソッドはrunまたはarunでLLMを使用するため、ここでは直接的な処理は不要です。
        raise NotImplementedError("このメソッドは使用されません。") 