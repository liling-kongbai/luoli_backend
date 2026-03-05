from logging import getLogger
from traceback import format_exc

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

logger = getLogger(__name__)


async def routine_graph_adapter_node(
    state, config: RunnableConfig, routine_graph: CompiledStateGraph
) -> dict:
    """常规层图适配器节点"""

    routine_state = {
        'messages': state.messages,
        'user_input_content': state.messages[-1].content,
        'introspect_count': 0,
        'response_draft_content': None,
    }

    try:
        routine_response = await routine_graph.ainvoke(routine_state, config)
    except Exception:
        logger.error(
            f'<routine_graph_adapter_node> 常规层图适配器节点报错！！！\n{format_exc()}'
        )
        raise

    return {'messages': [routine_response['messages'][-1]]}
