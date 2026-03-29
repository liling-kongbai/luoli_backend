from typing import Annotated

from langchain_core.messages.base import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import Field
from pydantic.main import BaseModel

from ..type import ExpandAction, FinalExecutePlan, LATSTreeNode


class MainGraphState(BaseModel):
    """主图状态"""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # 意图相关
    intent: str | None = Field(default=None)


class RoutineGraphState(BaseModel):
    """常规层图状态"""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    user_input_content: str | None = Field(default=None)  # 用户输入内容
    response_draft_content: str | None = Field(default=None)  # 响应草稿内容

    # 反思相关
    introspect_count: int = Field(default=0)
    introspection: str | None = Field(default=None)
    introspect_reason: str | None = Field(default=None)


# LATS 相关
def merge_tree_nodes(
    left: dict[str, LATSTreeNode], right: dict[str, LATSTreeNode]
) -> dict[str, LATSTreeNode]:
    """合并树节点"""

    return (left or {}) | (right or {})


class InferenceGraphState(BaseModel):
    """推理层图状态"""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    user_input_content: str | None = Field(default=None)  # 用户输入内容

    root_node_id: str
    current_node_id: str | None = Field(default=None)
    tree_nodes: Annotated[dict[str, LATSTreeNode], merge_tree_nodes] = Field(
        default_factory=dict
    )

    iterate_count: int = Field(default=0)
    llm_call_count: int = Field(default=0)

    candidates: list[ExpandAction] | None = Field(default=None)

    final_execute_plan: FinalExecutePlan | None = Field(default=None)
