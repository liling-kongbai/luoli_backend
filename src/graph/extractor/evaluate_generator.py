from ..type import EvaluateResult
from .base_structured_output_extractor import BaseStructuredOutputExtractor


class EvaluateGenerator(BaseStructuredOutputExtractor):
    """评估生成器"""

    OUTPUT_SCHEMA = EvaluateResult
    SYSTEM_PROMPT = """\
        你是一个极其严格的逻辑裁判员。你的任务是评估 Agent 执行的 **最新一步动作** 对实现 **原始终极目标** 的价值。

        **原始终极目标：**
        <<<
        {user_input_content}
        >>>

        **评估标准：**

        1. **评分（0 - 10）**：
        - 0 分：逻辑错误、工具报错且无修复建议、严重的幻觉、偏离目标。
        - 3-5 分：动作执行成功，但信息价值一般，或者只是“模拟”成功。
        - 6-8 分：获得了关键性的新信息，显著推进了进度。
        - 9-10 分：彻底解决了核心难题，或任务已完成。

        2. **死胡同判定**：
        - 如果工具返回 "Not Found"、"Error" 且换参数也无法解决，标记为 True。
        - 如果陷入循环，标记为 True。
        - 如果判断此动作无继续执行的价值，标记为 True。

        3. **完结判定**：
        - 只有当用户的所有要求都已得到满足（或已获得足够信息撰写最终报告）时，才标记为 True。
        - 注意：如果这一步只是 Mock（模拟执行），通常不能算作 completed，除非任务本身就是制定计划。
        - 只有当任务已经彻底完成时，彻底能够解决用户的初始目标时，才标记为 True。

        请注意，按照要求返回相关的内容，不要输出错误的格式，不要输出错误的内容，不要包含任何额外的解释或文本。
        """
