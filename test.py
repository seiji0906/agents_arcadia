import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tools.file_read_tool import FileReadTool
from agents.coding_agent import CodingAgent
from agents.planning_agent import PlanningAgent
from agents.review_agent import ReviewAgent
from agents.file_operation_agent import FileOperationAgent
from workflow import build_workflow
from langchain_core.messages import HumanMessage

def main():
    # .envファイルから環境変数を読み込む
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # ツールを準備
    file_read_tool = FileReadTool()
    tools = [file_read_tool]  # 必要に応じて追加

    # LLMを準備
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o",  # 適宜変更
        openai_api_key=OPENAI_API_KEY
    )

    # 各エージェントのインスタンスを作成
    coding_agent = CodingAgent(llm, tools)
    planning_agent = PlanningAgent(llm, tools)
    review_agent = ReviewAgent(llm, tools)
    file_operation_agent = FileOperationAgent(llm, tools)

    # ワークフロー構築
    graph = build_workflow()

    # 入力例
    inputs = {
        "messages": [
            HumanMessage(content="テキストファイルの中で、頻出する単語の上位3位までのランキングを作成する。無視すべき単語（例えば ‘the’, ‘and’ など）は除外してください。")
        ]
    }

    # Config
    config = {
        "configurable": {
            "coding_agent": coding_agent,
            "planning_agent": planning_agent,
            "review_agent": review_agent,
            "file_read_tool": file_read_tool,
            "file_operation_agent": file_operation_agent
        }
    }

    # 同期実行（ストリーム形式）
    for output in graph.stream(inputs, config):
        print(output)

    # 非同期実行を行う場合
    # import asyncio
    # async def run_async():
    #     async for output in graph.astream(inputs, config):
    #         print(output)
    # asyncio.run(run_async())

if __name__ == "__main__":
    main()