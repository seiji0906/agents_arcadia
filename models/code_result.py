"""
CodeResult: コードのファイルパスと実際のコードを格納するPydanticモデル
"""

from pydantic import BaseModel

class CodeResult(BaseModel):
    file_path: str
    code: str 