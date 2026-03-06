from typing import Annotated

from langchain_core.messages.base import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import Field
from pydantic.main import BaseModel


class MainGraphState(BaseModel):
    """主图状态"""

    messages: Annotated[list[BaseMessage], add_messages]

    intent: str | None = Field(default=None)


class RoutineGraphState(BaseModel):
    """常规层图状态"""

    messages: Annotated[list[BaseMessage], add_messages]
    user_input_content: str | None = Field(default=None)  # 用户输入内容
    introspect_count: int = Field(default=0)  # 反思计数
    response_draft_content: str | None = Field(default=None)  # 响应草稿内容

    introspection: str | None = Field(default=None)
    introspect_reason: str | None = Field(default=None)
