from langchain_core.runnables.config import RunnableConfig

from ..prompt import (
    ROUTINE_CHAT_SYSTEM_PROMPT_TEMPLATE,
    ROUTINE_FINAL_CHAT_SYSTEM_PROMPT_TEMPLATE,
)


async def routine_chat_node(state, config: RunnableConfig) -> dict:
    """常规层对话节点"""

    llm = config['configurable'].get('llm')

    if introspect_reason := state.introspect_reason:
        introspect_reason_prompt = f'上一轮结果经过反思评估后，未通过，请针对以下反馈进行改进：{introspect_reason}'
    else:
        introspect_reason_prompt = ''

    chain = ROUTINE_CHAT_SYSTEM_PROMPT_TEMPLATE | llm
    response = await chain.ainvoke(
        {
            'user_name': config['configurable'].get('user_name', '理灵'),
            'introspect_reason': introspect_reason_prompt,
            'messages': state.messages,
        },
        config,
    )
    return {
        'messages': [response],
        'introspect_reason': None,
    }


async def routine_final_chat_node(state, config: RunnableConfig) -> dict:
    """常规层最终对话节点"""

    llm = config['configurable'].get('llm')

    chain = ROUTINE_FINAL_CHAT_SYSTEM_PROMPT_TEMPLATE | llm
    response = await chain.ainvoke(
        {
            'user_input_content': state.user_input_content,
            'response_draft_content': state.response_draft_content,
        },
        config,
    )
    return {'messages': [response]}
