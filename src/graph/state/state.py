from typing import Annotated

from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import Field
from pydantic.main import BaseModel


class MainGraphState(BaseModel):
    """主图状态"""

    messages: Annotated[list[BaseMessage], add_messages]


class RoutineGraphState(BaseModel):
    """常规层图状态"""

    messages: Annotated[list[BaseMessage], add_messages]
    introspect_count: int = Field(default=0)  # 反思计数
    response_draft: AIMessage | None = Field(default=None)  # 响应草稿
