from ..type import FinalExecutePlan
from .base_structured_output_extractor import BaseStructuredOutputExtractor


class FinalExecutePlanGenerator(BaseStructuredOutputExtractor):
    """最终执行计划生成器"""

    OUTPUT_SCHEMA = FinalExecutePlan
    SYSTEM_PROMPT = """\
        你是一个任务规划大师。请根据经过深思熟虑（LATS 树搜索）选出的最优路径，整理出一份最终执行计划。
        你需要根据原始用户问题和最优处理路径，按照要求的格式规范生成一份最终执行计划。

        **原始用户问题：**
        <<<
        {user_input_content}
        >>>

        **最优处理路径：**
        <<<
        {trajectory}
        >>>

        **任务要求：**
        1. 分析这条路径上的每一步。
        2. 如果某个步骤是没有完成的，或是 **Mock/模拟执行** 的，标记为 `pending`。
        3. 如果某个步骤是 **真实执行** 过且成功的，标记为 `completed`，并提取结果。
        4. 有些字段需要填写，有些不需要或者返回 None。比如 result 字段，如果某个步骤是 **Mock/模拟执行** 的，就没有结果，要么不返回任何值，要么返回 None。

        请注意，按照要求返回相关的内容，不要输出错误的格式，不要输出错误的内容，不要包含任何额外的解释或文本。
        """
