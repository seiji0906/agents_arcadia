"""
ワークフローで使用するノード（read_code_node, planning_node, review_node, coding_node, apply_code_node 等）
"""
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from agents.coding_agent import CodingAgent
from agents.planning_agent import PlanningAgent
from agents.review_agent import ReviewAgent
from agents.file_operation_agent import FileOperationAgent
from agents.command_generation_agent import CommandGenerationAgent
from tools.file_read_tool import FileReadTool
from models.agent_state import AgentState
from models.code_result import CodeResult
from typing import Any, Optional
from pydantic import ValidationError
import json
import logging
from agents.terminal_agent import TerminalAgent
from agents.browser_agent import BrowserAgent

logging.basicConfig(level=logging.DEBUG)

def read_code_node(state: AgentState, config: RunnableConfig):
    """
    /generate フォルダ内の指定ファイルを読み込むノード
    """
    target_file_path = "generate/target.py"  # 修正対象のファイルパス（適宜変更可）

    file_read_tool: FileReadTool = config["configurable"].get("file_read_tool")
    if file_read_tool is None:
        raise ValueError("FileReadTool is not configured")

    file_content = file_read_tool.run(target_file_path)

    return {
        "existing_code": file_content,
        "target_file_path": target_file_path
    }

async def acoding_node(state: AgentState, config: RunnableConfig):
    """非同期版coding_node"""
    agent: CodingAgent = config["configurable"]["coding_agent"]
    messages = state["messages"]

    existing_code = state.get("existing_code", "")
    target_file_path = state.get("target_file_path", "generate/target.py")

    messages_to_pass = list(messages)
    messages_to_pass.append(
        HumanMessage(content=f"以下は {target_file_path} の現在の内容です:\n\n{existing_code}")
    )

    response_str = await agent.arun(messages_to_pass, config)

    return {
        "messages": [response_str],
        "coding_result": {
            "code": response_str,
            "file_path": target_file_path
        }
    }

def coding_node(state: AgentState, config: RunnableConfig):
    """同期版coding_node"""
    agent: CodingAgent = config["configurable"]["coding_agent"]
    messages = state["messages"]

    existing_code = state.get("existing_code", "")
    target_file_path = state.get("target_file_path", "generate/target.py")

    messages_to_pass = list(messages)
    messages_to_pass.append(
        HumanMessage(content=f"以下は {target_file_path} の現在の内容です:\n\n{existing_code}")
    )

    response_str = agent.run(messages_to_pass, config)

    return {
        "messages": [response_str],
        "coding_result": {
            "code": response_str,
            "file_path": target_file_path
        }
    }

def planning_node(state: AgentState, config: RunnableConfig):
    """同期版planning_node"""
    agent: PlanningAgent = config["configurable"]["planning_agent"]
    messages = state["messages"]
    response = agent.run(messages, config)
    return {"messages": [response], "requirements": response.content}

async def aplanning_node(state: AgentState, config: RunnableConfig):
    """非同期版planning_node"""
    agent: PlanningAgent = config["configurable"]["planning_agent"]
    messages = state["messages"]
    response = await agent.arun(messages, config)
    return {"messages": [response], "requirements": response.content}

def review_node(state: AgentState, config: RunnableConfig):
    """同期版review_node"""
    agent: ReviewAgent = config["configurable"]["review_agent"]
    messages = state["messages"]
    messages.append(
        HumanMessage(content=f"以下の要件をご確認ください:\n{state['requirements']}")
    )
    response = agent.run(messages, config)
    return {"messages": [response], "review_result": response.content}

async def areview_node(state: AgentState, config: RunnableConfig):
    """非同期版review_node"""
    agent: ReviewAgent = config["configurable"]["review_agent"]
    messages = state["messages"]
    messages.append(
        HumanMessage(content=f"以下の要件をご確認ください:\n{state['requirements']}")
    )
    response = await agent.arun(messages, config)
    return {"messages": [response], "review_result": response.content}

def file_operation_node(state: AgentState, config: RunnableConfig):
    """
    新設のファイル操作ノード。
    coding_node / acoding_node が生成した coding_result["code"] を受け取り、
    FileOperationAgent に渡して実際のファイル操作を行うか判断・実行する。
    """
    agent: FileOperationAgent = config["configurable"]["file_operation_agent"]
    coding_result = state.get("coding_result", {})
    raw_text = coding_result.get("code", "")
    file_path = coding_result.get("file_path")

    if not raw_text.strip():
        return {"file_operation_result": "コーディングエージェントの出力が空です。"}

    if not file_path:
        return {"file_operation_result": "ファイルパスが指定されていません。"}

    # 同期呼び出し
    result = agent.run(f'{{"file_path": "{file_path}", "code": "{raw_text}"}}', config)
    return {"file_operation_result": result}

async def afile_operation_node(state: AgentState, config: RunnableConfig):
    """非同期版ファイル操作ノード"""
    agent: FileOperationAgent = config["configurable"]["file_operation_agent"]
    coding_result = state.get("coding_result", {})
    raw_text = coding_result.get("code", "")
    file_path = coding_result.get("file_path")

    if not raw_text.strip():
        return {"file_operation_result": "コーディングエージェントの出力が空です。"}

    if not file_path:
        return {"file_operation_result": "ファイルパスが指定されていません。"}

    # 非同期呼び出し
    result = await agent.arun(f'{{"file_path": "{file_path}", "code": "{raw_text}"}}', config)
    return {"file_operation_result": result}

def should_continue(state: AgentState) -> str:
    """
    次にどのノードを呼び出すかを決定する関数。
    """
    if not state.get("requirements"):
        return "planning"
    elif not state.get("review_result"):
        return "review"
    else:
        return "coding" 

def command_generation_node(state: AgentState, config: RunnableConfig):
    """
    CommandGenerationAgent を呼び出すノード。
    """
    agent: CommandGenerationAgent = config["configurable"]["command_generation_agent"]
    messages = state.get("messages", [])
    file_operation_result = state.get("file_operation_result", "")

    logging.info(f"command_generation_node - state: {state}") # ステートの内容を出力

    messages_to_pass = list(messages)
    messages_to_pass.append(
        HumanMessage(content=f"ファイル操作の結果: {file_operation_result}。これに基づき、次に実行すべきコマンドを生成してください。")
    )

    command_json = agent.run(messages_to_pass, config)

    logging.info(f"command_generation_node - generated_command: {command_json}") # 生成されたコマンドを出力

    # JSON 文字列からコマンドを抽出
    try:
        command = json.loads(command_json)["command"]
    except (json.JSONDecodeError, KeyError):
        logging.error(f"command_generation_node - コマンドのパースに失敗しました。")
        return {
            "messages": messages_to_pass,
            "generated_command": "",  # コマンドを空にする
            "file_operation_result": file_operation_result
        }

    return_value = {
        "messages": messages_to_pass,
        "generated_command": command, # JSON ではなく、コマンド文字列を格納
        "file_operation_result": file_operation_result
    }
    print(f"command_generation_node - returning: {return_value}")
    return return_value

async def acommand_generation_node(state: AgentState, config: RunnableConfig):
    """
    CommandGenerationAgent の非同期呼び出しノード。
    """
    agent: CommandGenerationAgent = config["configurable"]["command_generation_agent"]
    messages = state.get("messages", [])
    file_operation_result = state.get("file_operation_result", "")

    logging.info(f"acommand_generation_node - state: {state}") # ステートの内容を出力

    messages_to_pass = list(messages)
    messages_to_pass.append(
        HumanMessage(content=f"ファイル操作の結果: {file_operation_result}。これに基づき、次に実行すべきコマンドを生成してください。")
    )

    command_json = await agent.arun(messages_to_pass, config)

    logging.info(f"acommand_generation_node - generated_command: {command_json}") # 生成されたコマンドを出力

    # JSON 文字列からコマンドを抽出
    try:
        command = json.loads(command_json)["command"]
    except (json.JSONDecodeError, KeyError):
        logging.error(f"acommand_generation_node - コマンドのパースに失敗しました。")
        return {
            "messages": messages_to_pass,
            "generated_command": "",  # コマンドを空にする
            "file_operation_result": file_operation_result
        }

    return {
        "messages": messages_to_pass,
        "generated_command": command, # JSON ではなく、コマンド文字列を格納
        "file_operation_result": file_operation_result
    }

def terminal_node(state: AgentState, config: RunnableConfig):
    """
    TerminalAgent を呼び出すノード。
    """
    agent: TerminalAgent = config["configurable"]["terminal_agent"]
    messages = state.get("messages", [])
    generated_command = state.get("generated_command", "") # JSON ではなく、文字列として取得

    logging.info(f"terminal_node - state: {state}") # ステートの内容を出力

    if not generated_command:
        logging.error(f"terminal_node - コマンドが生成されていません。")
        return {
            "messages": messages,
            "terminal_command": "コマンドが生成されていません。"
        }

    logging.info(f"terminal_node - 抽出されたコマンド: {generated_command}") # 抽出されたコマンドを出力

    # TerminalAgent.run にコマンドを文字列として渡す
    logging.info(f"terminal_node - TerminalAgent.run 呼び出し前の generated_command: {generated_command}") # 呼び出し前にコマンドを出力
    command_result = agent.run(generated_command) # config を削除
    logging.info(f"terminal_node - コマンド実行結果: {command_result}") # 実行結果を出力

    return {
        "messages": messages, # 以前のメッセージ履歴を引き続き渡す
        "terminal_command": command_result  # エラーメッセージ含む実行結果を格納
    }

async def aterminal_node(state: AgentState, config: RunnableConfig):
    """
    TerminalAgent の非同期呼び出しノード。
    """
    agent: TerminalAgent = config["configurable"]["terminal_agent"]
    messages = state.get("messages", [])
    generated_command = state.get("generated_command", "") # JSON ではなく、文字列として取得

    logging.info(f"aterminal_node - state: {state}") # ステートの内容を出力

    if not generated_command:
        logging.error(f"aterminal_node - コマンドが生成されていません。")
        return {
            "messages": messages,
            "terminal_command": "コマンドが生成されていません。"
        }

    logging.info(f"aterminal_node - 抽出されたコマンド: {generated_command}") # 抽出されたコマンドを出力

    # TerminalAgent.arun にコマンドを文字列として渡す
    logging.info(f"aterminal_node - TerminalAgent.arun 呼び出し前の generated_command: {generated_command}") # 呼び出し前にコマンドを出力
    command_result = await agent.arun(generated_command) # config を削除
    logging.info(f"aterminal_node - コマンド実行結果: {command_result}") # 実行結果を出力

    return {
        "messages": messages, # 以前のメッセージ履歴を引き続き渡す
        "terminal_command": command_result  # エラーメッセージ含む実行結果を格納
    }

def browser_node(state: AgentState, config: RunnableConfig):
    """
    BrowserAgent を呼び出すノード。
    """
    agent: BrowserAgent = config["configurable"]["browser_agent"]
    messages = state.get("messages", [])
    # ここでは任意の入力を想定
    result = agent.run(messages, config)
    return {
        "messages": messages,
        "browser_result": result
    }

async def abrowser_node(state: AgentState, config: RunnableConfig):
    """
    BrowserAgent の非同期呼び出しノード。
    """
    agent: BrowserAgent = config["configurable"]["browser_agent"]
    messages = state.get("messages", [])
    result = await agent.arun(messages, config)
    return {
        "messages": messages,
        "browser_result": result
    }