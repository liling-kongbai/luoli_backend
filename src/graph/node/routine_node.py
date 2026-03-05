from logging import getLogger

from langchain_core.runnables.config import RunnableConfig

from ..prompt import ROUTINE_FINAL_CHAT_SYSTEM_PROMPT_TEMPLATE

logger = getLogger(__name__)


async def routine_final_chat_node(state, config: RunnableConfig) -> dict:
    """常规层最终对话节点"""

    llm = config['configurable'].get('llm')

    chain = ROUTINE_FINAL_CHAT_SYSTEM_PROMPT_TEMPLATE | llm
    response = await chain.ainvoke(
        {
            'user_input_content': state.user_input_content,
            'response_draft_content': state.response_draft_content,
            'input': state.user_input_content,
        },
        config,
    )
    return {'messages': [response]}
