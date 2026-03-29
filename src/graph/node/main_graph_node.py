from logging import getLogger
from traceback import format_exc

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ..state import InferenceGraphState, MainGraphState, RoutineGraphState
from ..type import LATSTreeNode

logger = getLogger(__name__)


async def routine_graph_adapter_node(
    state: MainGraphState, config: RunnableConfig, routine_graph: CompiledStateGraph
) -> dict:
    """常规层图适配器节点"""

    user_input_content = ''
    if final_execute_plan := state.final_execute_plan:
        user_input_content = f'请根据用户的原始目标/问题和完整的执行计划，完成/回答用户的要求/问题。\n用户原始问题：{final_execute_plan.original_goal}\n完整执行计划：\n'
        for step in final_execute_plan.steps:
            user_input_content += f'步骤 {step.id} ：\n思考：{step.thought}\n需要调用的工具的名称：{step.tool_name}\n需要调用的工具的参数：{step.tool_args}\n工具的运行结果：{step.result}\n此步骤是否完成：{step.status.value}\n'

    routine_graph_state = RoutineGraphState(
        {
            'messages': state.messages,
            'user_input_content': user_input_content
            if user_input_content
            else state.messages[-1].content,
            'response_draft_content': None,
            'introspect_count': 0,
            'introspection': None,
            'introspect_reason': None,
        }
    )

    try:
        response = await routine_graph.ainvoke(routine_graph_state, config)
    except Exception:
        logger.error(
            f'<routine_graph_adapter_node> 常规层图适配器节点报错！！！\n{format_exc()}'
        )
        raise
    return {'messages': [response['messages'][-1]], 'final_execute_plan': None}


async def inference_graph_adapter_node(
    state: MainGraphState, config: RunnableConfig, inference_graph: CompiledStateGraph
) -> dict:
    """推理层图适配器节点"""

    root_node = LATSTreeNode(
        parent_id=None,
        child_ids=None,
        depth=0,
        summary_generate_depth=0,
        summary=None,
        visit_count=0,
        score_count=0.0,
        action=None,
        observation=None,
        is_completed=False,
        is_pruned=False,
        pruned_reason=None,
    )
    root_node_id = root_node.id
    inference_graph_state = InferenceGraphState(
        {
            'messages': state.messages,
            'user_input_content': state.messages[-1].content,
            'root_node_id': root_node_id,
            'current_node_id': root_node_id,
            'tree_nodes': {root_node_id: root_node},
            'iterate_count': 0,
            'llm_call_count': 0,
            'candidates': None,
            'final_execute_plan': None,
        }
    )

    try:
        response = await inference_graph.ainvoke(inference_graph_state, config)
    except Exception:
        logger.error(
            f'<inference_graph_adapter_node> 推理层图适配器节点报错！！！\n{format_exc()}'
        )
        raise
    return {'final_execute_plan': response['final_execute_plan']}
