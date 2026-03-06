from logging import getLogger
from traceback import format_exc

from langchain_core.runnables.config import RunnableConfig

from ..type import IntentClassification, IntrospectionClassification
from .structured_output_extractor import IntentClassifier, IntrospectClassifier

logger = getLogger(__name__)


# 意图相关
async def intent_classifier_node(state, config: RunnableConfig) -> dict:
    """意图分类器节点"""

    llm = config['configurable'].get('llm')

    try:
        chain = IntentClassifier(llm).get_extractor_chain()
        result = await chain.ainvoke(
            {
                'messages': state.messages[-10:],
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


def intent_classifier_condition(state) -> str:
    """意图分类器条件"""

    return state.intent


# 反思相关
async def introspect_classifier_node(state, config: RunnableConfig) -> dict:
    """反思分类器节点"""

    introspect_count = state.introspect_count
    if introspect_count >= 3:
        logger.warning(
            '<introspect_classifier_node> 反思次数超过 3 次，直接返回最终响应层！！！'
        )
        return {'introspection': IntrospectionClassification.FinalChatLayer.value}

    llm = config['configurable'].get('llm')

    try:
        chain = IntrospectClassifier(llm).get_extractor_chain()
        result = await chain.ainvoke(
            {
                'messages': state.messages,
                'response_draft': state.response_draft.content
                if state.response_draft
                else '暂时没有响应草稿',
                'input': f'本次用户的消息/问题：{state.user_input_content}',
            },
            config,
        )

        if (
            result.introspection.value
            == IntrospectionClassification.FinalChatLayer.value
        ):
            return {'introspection': result.introspection.value}
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
        return {'introspection': IntrospectionClassification.FinalChatLayer.value}


def introspect_classifier_condition(state) -> str:
    """反思分类器条件"""

    return state.introspection
