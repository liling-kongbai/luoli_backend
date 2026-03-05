from logging import getLogger
from traceback import format_exc

from langchain_core.runnables.config import RunnableConfig

from ..type import IntentClassification, IntrospectionClassification
from .structured_output_extractor import IntentClassifier, IntrospectClassifier

logger = getLogger(__name__)


async def intent_classifier_condition(state, config: RunnableConfig) -> str:
    """意图分类器条件"""

    llm = config['configurable'].get('llm')
    if not llm:
        logger.error('<intent_classifier_condition> 当前无 LLM，请检查！！！')
        raise ValueError('<intent_classifier_condition> 当前无 LLM，请检查！！！')

    try:
        chain = IntentClassifier(llm).get_extractor_chain()
        result = await chain.ainvoke(
            {
                'messages': state.messages[-10:-2],
                'input': state.messages[-1].content,
            },
            config,
        )
        return result.intent.value
    except Exception:
        logger.error(
            f'<intent_classifier_condition> 意图分类器条件报错！！！\n{format_exc()}'
        )
        return IntentClassification.RoutineLayer.value


async def introspect_classifier_condition(state, config: RunnableConfig) -> str:
    """反思分类器条件"""

    llm = config['configurable'].get('llm')
    if not llm:
        logger.error('<introspect_classifier_condition> 当前无 LLM，请检查！！！')
        raise ValueError('<introspect_classifier_condition> 当前无 LLM，请检查！！！')

    if state.introspect_count >= 3:
        logger.warning(
            '<introspect_classifier_condition> 反思次数超过 3 次，直接返回最终响应层！！！'
        )
        state.introspect_count = 0
        return IntrospectionClassification.FinalResponseLayer.value

    try:
        chain = IntrospectClassifier(llm).get_extractor_chain()
        result = await chain.ainvoke(
            {
                'messages': state.messages[-10:-2],
                'response_draft': state.response_draft.content,
                'input': state.response_draft.content,
            },
            config,
        )
        return result.introspection.value
    except Exception:
        logger.error(
            f'<introspect_classifier_condition> 反思分类器条件报错！！！\n{format_exc()}'
        )
        return IntrospectionClassification.FinalResponseLayer.value
