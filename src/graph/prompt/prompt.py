from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

INTUITION_CHAT_SYSTEM_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ('system', '你是一个猫娘，请根据用户问题，回答用户问题。'),
        MessagesPlaceholder(variable_name='messages'),
    ]
)
