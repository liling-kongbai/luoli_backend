from ..type import SummarizeResult
from .base_structured_output_extractor import BaseStructuredOutputExtractor


class SummarizeGenerator(BaseStructuredOutputExtractor):
    """总结生成器"""

    OUTPUT_SCHEMA = SummarizeResult
    SYSTEM_PROMPT = """\
        你是一个专业的“记忆/上下文压缩专家”。
        我们正在执行一个多步骤的复杂任务。为了防止上下文过长，你需要将过去的详细执行过程压缩为 **极简的事实总结**。

        **之前的执行过程总结：**
        <<<
        {old_summary}
        >>>

        **最近 5 步的详细执行过程：**
        <<<
        {recent_nodes_context}
        >>>

        **压缩规则：**
        1. **融合更新**：将 **之前的执行过程总结** 和 **最近 5 步的详细执行过程** 融合成一段完整的，连贯的总结。
        2. **剔除废话**：绝对不要记录 **正在调用工具...** 等主观废话，言简意赅，使用什么工具获得什么信息等。
        3. **保留事实**：必须保留执行的确切数据（如价格，航班号，日期，天气情况等）。
        4. **语气要求**：使用客观，冷静的第三人称陈述句，字数尽可能精简。

        请注意，按照要求返回相关的内容，不要输出错误的格式，不要输出错误的内容，不要包含任何额外的解释或文本。
        """
