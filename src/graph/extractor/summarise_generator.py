from ..type import SummariseResult
from .base_structured_output_extractor import BaseStructuredOutputExtractor


class SummariseGenerator(BaseStructuredOutputExtractor):
    """总结生成器"""

    OUTPUT_SCHEMA = SummariseResult
    SYSTEM_PROMPT = """\
        你是一个专业的“记忆压缩专家”。
        我们在执行一个多步骤的复杂任务。为了防止上下文过长，你需要将大模型过去的**详细执行过程**压缩为**极简的事实摘要**。

        **之前的执行过程摘要**
        {old_summary}

        **最近 5 步的详细操作历史**
        {recent_action_history}

        **压缩规则：**
        1. **融合更新**：将**之前的执行过程摘要**和**最近 5 步的详细操作历史**融合成一段完整的，最新的，连贯的摘要。
        2. **剔除废话**：绝对不要记录**我正在调用工具...**，**大模型思考认为...**等主观废话，直接言简意赅，使用什么工具获得什么信息等核心事实。
        3. **保留核心事实**：必须保留执行的确切数据（如价格，航班号，日期，天气情况，特定的报错代码等）。
        4. **语气要求**：使用客观，冷静的第三人称陈述句，字数尽可能精简。

        请注意，按照要求返回相关的内容，不要输出错误的格式，不要输出错误的内容，不要包含任何额外的解释或文本。
        """
