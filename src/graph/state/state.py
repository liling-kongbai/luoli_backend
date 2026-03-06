from typing import Annotated

from langchain_core.messages.base import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import Field
from pydantic.main import BaseModel


class MainGraphState(BaseModel):
    """主图状态"""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=[])

    # 意图相关
    intent: str | None = Field(default=None)


class RoutineGraphState(BaseModel):
    """常规层图状态"""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=[])
    user_input_content: str | None = Field(default=None)  # 用户输入内容
    response_draft_content: str | None = Field(default=None)  # 响应草稿内容

    # 反思相关
    introspect_count: int = Field(default=0)
    introspection: str | None = Field(default=None)
    introspect_reason: str | None = Field(default=None)
