from typing import Any

from ...type import Introspection, IntrospectionClassification
from .base_structured_output_extractor import BaseStructuredOutputExtractor


class IntrospectClassifier(BaseStructuredOutputExtractor):
    """反思分类器"""

    OUTPUT_SCHEMA = Introspection
    SYSTEM_PROMPT = """\
        你是一个专业的 AI 回复评估员。你的任务是基于部分对话历史，对 AI 助手的回复草稿进行多维度打分，并根据你的分数做出最终决策。

        **部分对话历史:**
        <<<
        {messages}
        >>>

        **AI 助手的回复草稿:**
        <<<
        {response_draft}
        >>>

        **评估指南:**
        请从以下维度进行 1 - 5 分的评分（1 = 差，3 = 合格，5 = 优秀）。
            1. **正确性**: 回复是否准确、真实地解决了用户的核心问题？
            2. **流畅性**: 回复是否流畅、自然、符合上下文？
            3. 如果只是普通的打招呼或聊天对话，只要没有出现什么问题，两个维度可以直接给 3 分以上。

        **决策指南:**
        - 如果 **正确性** 或 **流畅性** 评分 **都低于 3 分**，说明回复有重大缺陷，最终决策应该输出“{IntrospectLayer}”，（重试）。
        - 如果 **正确性** 和 **流畅性** 评分 **都达到或超过 3 分**，说明回复质量足够高，最终决策应该输出“{FinalResponseLayer}”，（接受）。

        请注意，按照要求返回相关的内容，不要输出错误的格式，不要输出错误的内容，不要包含任何额外的解释或文本。
        """

    def _get_partial_variables(self) -> dict[str, Any]:
        return {
            'IntrospectLayer': IntrospectionClassification.IntrospectLayer.value,
            'FinalResponseLayer': IntrospectionClassification.FinalResponseLayer.value,
        }
