from logging import getLogger
from traceback import format_exc

from langchain_core.runnables.config import RunnableConfig

from ..type import IntentClassification, IntrospectionClassification
from .structured_output_extractor import IntentClassifier, IntrospectClassifier

logger = getLogger(__name__)


async def intent_classifier_node(state, config: RunnableConfig) -> dict:
    """意图分类器节点"""

    llm = config['configurable'].get('llm')

    try:
        chain = IntentClassifier(llm).get_extractor_chain()
        result = await chain.ainvoke(
            {
                'messages': state.messages[-10:-2],
                'input': state.messages[-1].content,
            },
            config,
        )
        return {'intent': result.intent.value}
    except Exception:
        logger.error(
            f'<intent_classifier_node> 意图分类器节点报错！！！\n{format_exc()}'
        )
        return {'intent': IntentClassification.RoutineLayer.value}


async def introspect_classifier_node(state, config: RunnableConfig) -> dict:
    """反思分类器节点"""

    introspect_count = state.introspect_count
    if introspect_count >= 3:
        logger.warning(
            '<introspect_classifier_node> 反思次数超过 3 次，直接返回最终响应层！！！'
        )
        return {
            'introspect_count': 0,
            'introspection': IntrospectionClassification.FinalResponseLayer.value,
            'introspect_reason': None,
        }

    llm = config['configurable'].get('llm')

    try:
        chain = IntrospectClassifier(llm).get_extractor_chain()
        result = await chain.ainvoke(
            {
                'messages': state.messages[-10:-2],
                'response_draft': state.response_draft.content
                if state.response_draft
                else '暂时没有响应草稿',
                'input': '上一轮响应结果未通过原因：'
                + (state.introspect_reason or '无'),
            },
            config,
        )

        if (
            result.introspection.value
            == IntrospectionClassification.FinalResponseLayer.value
        ):
            return {
                'introspect_count': 0,
                'introspection': result.introspection.value,
                'introspect_reason': None,
            }
        else:
            return {
                'introspect_count': introspect_count + 1,
                'introspection': result.introspection.value,
                'introspect_reason': result.reason,
            }
    except Exception:
        logger.error(
            f'<introspect_classifier_node> 反思分类器节点报错！！！\n{format_exc()}'
        )
        return {
            'introspect_count': 0,
            'introspection': IntrospectionClassification.FinalResponseLayer.value,
            'introspect_reason': None,
        }
