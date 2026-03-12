from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import Field
from pydantic.main import BaseModel


# 意图相关
class IntentClassification(str, Enum):
    """枚举，意图类别"""

    IntuitionLayer = 'intuition_layer'  # 直觉层
    RoutineLayer = 'routine_layer'  # 常规层
    InferenceLayer = 'inference_layer'  # 推理层


class Intent(BaseModel):
    """数据模型，意图"""

    intent: IntentClassification


# 反思相关
class IntrospectionClassification(str, Enum):
    """枚举，反思类别"""

    Introspect = 'introspect'  # 反思
    Finalise = 'finalise'  # 最终


class Introspection(BaseModel):
    """数据模型，反思"""

    introspection: IntrospectionClassification
    reason: str | None = Field(default=None)


# 选择相关
class SelectionClassification(str, Enum):
    """枚举，选择类别"""

    Expand = 'expand'  # 扩展
    Summarise = 'summarise'  # 总结
    Finalise = 'finalise'  # 最终


class Selection(BaseModel):
    """数据模型，选择"""

    selection: SelectionClassification


# 反向传播相关
class BackpropagationClassification(str, Enum):
    """枚举，反向传播类别"""

    Select = 'select'  # 选择
    Finalise = 'finalise'  # 最终


class Backpropagation(BaseModel):
    """数据模型，反向传播"""

    backpropagation: BackpropagationClassification


# LATS 相关
class ExpandAction(BaseModel):
    """数据模型，扩展行动"""

    thought: str = Field(
        ..., description='思考过程（为什么选择使用这个工具和这些参数？意图是什么？）'
    )
    tool_name: str | None = Field(
        default=None,
        description='需要调用的工具名称（如果没有合适的工具，请返回 None）',
    )
    tool_args: dict[str, Any] | None = Field(
        default=None,
        description='需要调用的工具的参数（如果没有合适的工具或此工具不需要参数，请返回 None）',
    )


class ExpandResult(BaseModel):
    """数据模型，扩展结果"""

    candidates: list[ExpandAction] = Field(
        default_factory=list,
        min_length=0,
        max_length=3,
        description='生成 0 到 3 个截然不同的行动方案。如果认为任务已经彻底完成，请返回空列表',
    )


class LATSTreeNode(BaseModel):
    """数据模型，LATS 树节点"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    parent_id: str | None = Field(default=None)
    child_ids: list[str] = Field(default_factory=list)
    depth: int = Field(default=0)
    summary: str | None = Field(default=None)

    # UCT 相关
    visit_count: int = Field(default=0)
    score_count: float = Field(default=0.0)

    # 动作相关
    action: ExpandAction | None = Field(default=None)
    observation: str | None = Field(default=None)

    # 状态相关
    is_completed: bool = Field(default=False)
    is_pruned: bool = Field(default=False)
    pruned_reason: str | None = Field(default=None)
