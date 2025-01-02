"""
StateGraphの定義と各ノードの紐付けを行う。
"""
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from models.agent_state import AgentState
from nodes.nodes import (
    read_code_node,
    planning_node,
    aplanning_node,
    review_node,
    areview_node,
    coding_node,
    acoding_node,
    file_operation_node,
    should_continue,
    afile_operation_node,
    terminal_node,
    aterminal_node,
    browser_node,
    abrowser_node,
    command_generation_node,
    acommand_generation_node
)

def build_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    # ノードを追加
    workflow.add_node("read_code", RunnableLambda(read_code_node))
    workflow.add_node("planning", RunnableLambda(planning_node, afunc=aplanning_node))
    workflow.add_node("review", RunnableLambda(review_node, afunc=areview_node))
    workflow.add_node("coding", RunnableLambda(coding_node, afunc=acoding_node))
    workflow.add_node("file_operation", RunnableLambda(file_operation_node, afunc=afile_operation_node))
    workflow.add_node("terminal", RunnableLambda(terminal_node, afunc=aterminal_node))
    workflow.add_node("browser", RunnableLambda(browser_node, afunc=abrowser_node))
    workflow.add_node("command_generation", RunnableLambda(command_generation_node, afunc=acommand_generation_node))

    # エントリーポイント
    workflow.set_entry_point("read_code")
    workflow.add_edge("read_code", "planning")

    # 条件付きエッジ（planning）
    workflow.add_conditional_edges(
        "planning",
        should_continue,
        {
            "planning": "planning",
            "review": "review",
            "coding": "coding"
        }
    )

    # 条件付きエッジ（review）
    workflow.add_conditional_edges(
        "review",
        lambda state: "planning" if "Please revise" in state.get("review_result", "") else "coding",
        {
            "planning": "planning",
            "coding": "coding"
        }
    )

    # コーディングエージェント → ファイル操作エージェント
    workflow.add_edge("coding", "file_operation")

    # ファイル操作エージェント → コマンド生成エージェント
    workflow.add_edge("file_operation", "command_generation")

    # コマンド生成エージェント → ターミナルエージェント
    workflow.add_edge("command_generation", "terminal")

    # ターミナルエージェント → ブラウザエージェント
    workflow.add_edge("terminal", "browser")

    # ブラウザエージェント → 終了
    workflow.add_edge("browser", END)

    # 終了点
    workflow.set_finish_point("browser")

    graph = workflow.compile()
    return graph