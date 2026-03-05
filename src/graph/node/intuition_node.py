from logging import getLogger

from langchain_core.messages.ai import AIMessage
from langchain_core.runnables.config import RunnableConfig

from ..prompt import INTUITION_CHAT_SYSTEM_PROMPT_TEMPLATE

logger = getLogger(__name__)


async def intuition_chat_node(state, config: RunnableConfig) -> dict:
    """直觉对话节点"""

    llm = config['configurable'].get('llm')
    if not llm:
        logger.warning('<intuition_chat_node> 当前无 LLM，请检查！！！')
        return {
            'messages': [
                AIMessage(content='<intuition_chat_node> 当前无 LLM，请检查！！！')
            ]
        }

    chain = INTUITION_CHAT_SYSTEM_PROMPT_TEMPLATE | llm
    response = await chain.ainvoke(
        {
            'user_name': config['configurable'].get('user_name', '用户'),
            'messages': state.messages,
        },
        config,
    )
    return {'messages': [response]}
