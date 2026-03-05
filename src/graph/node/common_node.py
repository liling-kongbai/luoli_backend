from langchain_core.runnables.config import RunnableConfig
from langgraph.prebuilt.tool_node import ToolNode


async def tools_node(state, config: RunnableConfig):
    """工具节点"""

    return await ToolNode(config['configurable'].get('tools')).ainvoke(state, config)
