from enum import Enum

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

    IntrospectLayer = 'introspect_layer'  # 反思层
    FinalChatLayer = 'final_chat_layer'  # 最终对话层


class Introspection(BaseModel):
    """数据模型，反思"""

    introspection: IntrospectionClassification
    reason: str | None = Field(default=None)  # 原因
