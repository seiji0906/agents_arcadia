"""
AgentState: エージェントの状態を表すTypedDict
"""

from typing import Optional, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing import Annotated

from typing import TypedDict

class AgentState(TypedDict):
    """エージェントの状態を表現するTypedDict"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    requirements: Optional[str]        # 要件定義
    review_result: Optional[str]       # レビュー結果
    coding_result: Optional[dict]      # コーディング結果（ファイルパスとコードを含む）
    existing_code: Optional[str]       # 既存のコードを保持
    target_file_path: Optional[str]    # 対象のファイルパス 