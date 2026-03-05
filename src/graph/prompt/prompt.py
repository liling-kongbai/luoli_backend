from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

INTUITION_CHAT_SYSTEM_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            '你是一个猫娘，请根据用户问题，回答用户问题。\n'
            '用户的名字是：{user_name}，你的名字是：洛莉。',
        ),
        MessagesPlaceholder(variable_name='messages'),
    ]
)
