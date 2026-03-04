from langchain_core.messages.ai import AIMessage
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.config import RunnableConfig


async def intuition_chat_node(state, config: RunnableConfig) -> dict:
    """直觉对话节点"""

    llm = config['configurable'].get('llm')
    if not llm:
        return {
            'messages': [
                AIMessage(content='<intuition_chat_node> 当前无 LLM，请检查！！！')
            ]
        }

    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            ('system', '{system_prompt}'),
            MessagesPlaceholder(variable_name='messages'),
        ]
    )
    chain = chat_prompt_template | llm
    response = await chain.ainvoke(
        {
            'system_prompt': state.system_prompt,
            'messages': state.messages,
        },
        config,
    )
    return {'messages': [response]}
