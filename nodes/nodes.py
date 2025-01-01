"""
ワークフローで使用するノード（read_code_node, planning_node, review_node, coding_node, apply_code_node 等）
"""
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from agents.coding_agent import CodingAgent
from agents.planning_agent import PlanningAgent
from agents.review_agent import ReviewAgent
from agents.file_operation_agent import FileOperationAgent
from tools.file_read_tool import FileReadTool
from models.agent_state import AgentState
from models.code_result import CodeResult
from typing import Any, Optional
from pydantic import ValidationError
import json

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
        HumanMessage(content=f"Here is the current content of {target_file_path}:\n\n{existing_code}")
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
        HumanMessage(content=f"Here is the current content of {target_file_path}:\n\n{existing_code}")
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
        HumanMessage(content=f"Please review the following requirements:\n{state['requirements']}")
    )
    response = agent.run(messages, config)
    return {"messages": [response], "review_result": response.content}

async def areview_node(state: AgentState, config: RunnableConfig):
    """非同期版review_node"""
    agent: ReviewAgent = config["configurable"]["review_agent"]
    messages = state["messages"]
    messages.append(
        HumanMessage(content=f"Please review the following requirements:\n{state['requirements']}")
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