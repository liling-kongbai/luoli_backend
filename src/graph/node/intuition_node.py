from langchain_core.runnables.config import RunnableConfig

from ..prompt import INTUITION_CHAT_SYSTEM_PROMPT_TEMPLATE


async def intuition_chat_node(state, config: RunnableConfig) -> dict:
    """直觉层对话节点"""

    llm = config['configurable'].get('llm')

    chain = INTUITION_CHAT_SYSTEM_PROMPT_TEMPLATE | llm
    response = await chain.ainvoke(
        {
            'user_name': config['configurable'].get('user_name', '用户'),
            'messages': state.messages,
        },
        config,
    )
    return {'messages': [response]}
