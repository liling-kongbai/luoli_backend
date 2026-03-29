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
    Finalize = 'finalize'  # 最终


class Introspection(BaseModel):
    """数据模型，反思"""

    introspection: IntrospectionClassification
    reason: str | None = Field(default=None)


# 选择相关
class SelectionClassification(str, Enum):
    """枚举，选择类别"""

    Expand = 'expand'  # 扩展
    Summarize = 'summarize'  # 总结
    Finalize = 'finalize'  # 最终


class Selection(BaseModel):
    """数据模型，选择"""

    selection: SelectionClassification


# 反向传播相关
class BackpropagationClassification(str, Enum):
    """枚举，反向传播类别"""

    Select = 'select'  # 选择
    Finalize = 'finalize'  # 最终


class Backpropagation(BaseModel):
    """数据模型，反向传播"""

    backpropagation: BackpropagationClassification


# 扩展相关
class ExpandAction(BaseModel):
    """数据模型，扩展行动"""

    thought: str = Field(
        ..., description='思考（为什么选择使用这个工具和这些参数？意图是什么？）'
    )
    tool_name: str | None = Field(
        default=None,
        description='需要调用的工具的名称（如果没有合适的工具，请返回 None）',
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
        description='生成 0 到 3 个截然不同的行动方案。如果认为任务已经彻底完成或进入死胡同无继续意义，请返回空列表',
    )


# 总结相关
class SummarizeResult(BaseModel):
    """数据模型，总结结果"""

    summary: str = Field(..., description='总结内容')


# 评估相关
class EvaluateResult(BaseModel):
    """数据模型，评估结果"""

    analysis: str = Field(
        ...,
        description='评估分析：这个结果对解决目标有帮助吗？好在哪里？不好在哪里？正确在哪里？错误在哪里？',
    )
    score: float = Field(..., ge=0.0, le=10.0, description='评估分数')
    is_completed: bool = Field(
        ...,
        description='如果当前情况已经足以解决用户的目标，则为 True；否则为 False',
    )
    is_pruned: bool = Field(
        ...,
        description='如果发现这是一条走不通的死胡同，或无继续意义，或评估分数为 0，则为 True；否则为 False',
    )


# 最终执行计划相关
class ActionStatus(str, Enum):
    """枚举，行动状态"""

    Completed = 'completed'  # 已完成
    Pending = 'pending'  # 待处理


class FinalExecutePlanStep(BaseModel):
    """数据模型，最终执行计划步骤"""

    id: int = Field(..., description='步骤编号/顺序')
    thought: str | None = Field(..., description='思考')
    tool_name: str | None = Field(default=None, description='工具名称')
    tool_args: dict[str, Any] | None = Field(default=None, description='工具参数')
    result: str | None = Field(default=None, description='结果')
    status: ActionStatus = Field(...)


class FinalExecutePlan(BaseModel):
    """数据模型，最终执行计划"""

    original_goal: str = Field(..., description='原始目标')
    steps: list[FinalExecutePlanStep] = Field(
        ..., description='按步骤/顺序排列的计划步骤'
    )


# LATS 相关
class LATSTreeNode(BaseModel):
    """数据模型，LATS 树节点"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    parent_id: str | None = Field(default=None)
    child_ids: list[str] = Field(default_factory=list)
    depth: int = Field(default=0)
    summary_generate_depth: int = Field(default=0)
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
