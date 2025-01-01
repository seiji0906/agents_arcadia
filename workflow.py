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
    afile_operation_node
)

def build_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    # ノードを追加
    workflow.add_node("read_code", RunnableLambda(read_code_node))
    workflow.add_node("planning", RunnableLambda(planning_node, afunc=aplanning_node))
    workflow.add_node("review", RunnableLambda(review_node, afunc=areview_node))
    workflow.add_node("coding", RunnableLambda(coding_node, afunc=acoding_node))
    workflow.add_node("file_operation", RunnableLambda(file_operation_node, afunc=afile_operation_node))

    # エントリーポイント
    workflow.set_entry_point("read_code")
    workflow.add_edge("read_code", "planning")

    # 条件付きエッジ
    workflow.add_conditional_edges(
        "planning",
        should_continue,
        {
            "planning": "planning",
            "review": "review",
            "coding": "coding"
        }
    )

    workflow.add_conditional_edges(
        "review",
        lambda state: "planning" if "Please revise" in state.get("review_result", "") else "coding",
        {
            "planning": "planning",
            "coding": "coding"
        }
    )

    # ここで coding -> file_operation と繋ぐ (apply_code_node を使わない場合)
    workflow.add_edge("coding", "file_operation")
    workflow.add_edge("file_operation", END)

    # 終了点をモジュールにセット
    workflow.set_finish_point("file_operation")

    graph = workflow.compile()
    return graph 