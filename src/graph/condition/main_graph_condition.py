from logging import getLogger
from traceback import format_exc

from langchain_core.runnables import RunnableConfig

from ..type import IntentClassification
from .structured_output_extractor import IntentClassifier

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
                'input': '\n'.join(
                    [
                        f'{message.type}: {message.content}'
                        for message in state.messages[-5:]
                    ]
                )
            },
            config,
        )
        return result.intent.value
    except Exception:
        logger.error(
            f'<intent_classifier_condition> 意图分类器条件报错！！！\n{format_exc()}'
        )
        return IntentClassification.RoutineLayer.value
