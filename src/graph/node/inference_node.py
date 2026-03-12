from asyncio.tasks import gather
from json import dumps
from logging import getLogger
from math import log, sqrt
from traceback import format_exc

from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.runnables.config import RunnableConfig

from ...manager import ToolManager
from ..extractor import ExpandGenerator
from ..state import InferenceGraphState
from ..type import ExpandAction, LATSTreeNode, SelectionClassification

logger = getLogger(__name__)


def compute_uct_score(
    child_visit_count: int,
    child_score_count: float,
    parent_visit_count: int,
    exploration_c: float = 1.414,
) -> float:
    """计算 UCT 分数"""

    if child_visit_count == 0:
        return float('inf')

    exploitation_score = child_score_count / child_visit_count  # 利用分数
    exploration_score = exploration_c * sqrt(
        log(parent_visit_count) / child_visit_count
    )  # 探索分数
    return exploitation_score + exploration_score


def select_best_leaf_node(
    root_id: str, tree_nodes: dict[str, LATSTreeNode], exploration_c: float = 1.414
) -> str:
    """选择最佳叶子节点"""

    current_node_id = root_id
    while True:
        current_node = tree_nodes[current_node_id]
        if current_node.is_completed or not current_node.child_ids:
            return current_node_id

        valid_child_nodes = []
        for child_id in current_node.child_ids:
            child_node = tree_nodes[child_id]
            if child_node and not child_node.is_pruned:
                valid_child_nodes.append(child_node)

        if not valid_child_nodes:
            return current_node_id

        best_leaf_node = None
        best_uct_score = -float('inf')
        for child_node in valid_child_nodes:
            uct_score = compute_uct_score(
                child_node.visit_count,
                child_node.score_count,
                current_node.visit_count,
                exploration_c,
            )
            if uct_score > best_uct_score:
                best_uct_score = uct_score
                best_leaf_node = child_node
        current_node_id = best_leaf_node.id


def selector_node(state: InferenceGraphState, config: RunnableConfig) -> dict:
    """选择器节点"""

    if state.iterate_count > config['configurable'].get('max_iterate_count'):
        logger.warning('<selector_node> 迭代计数超过最大迭代计数，停止推理！！！')
        return {'current_node_id': SelectionClassification.Finalise.value}

    try:
        return {
            'current_node_id': select_best_leaf_node(
                state.root_id,
                state.tree_nodes,
                config['configurable'].get('exploration_c'),
            )
        }
    except Exception:
        logger.error(f'<selector_node> 选择器节点报错！！！\n{format_exc()}')
        return {'current_node_id': SelectionClassification.Finalise.value}


def get_context_nodes_trajectory(
    tree_nodes: dict[str, LATSTreeNode], leaf_node_id: str
) -> list[LATSTreeNode]:
    """获取上下文节点轨迹"""

    context_nodes_trajectory = []
    current_node_id = leaf_node_id
    while current_node_id:
        current_node = tree_nodes[current_node_id]
        context_nodes_trajectory.append(current_node)
        current_node_id = current_node.parent_id
    return context_nodes_trajectory[::-1]


async def expander_node(state: InferenceGraphState, config: RunnableConfig) -> dict:
    """扩展器节点"""

    current_node_id = state.current_node_id
    tree_nodes = state.tree_nodes
    current_node = tree_nodes[current_node_id]
    current_node_depth = current_node.depth
    recent_depth = current_node_depth % config['configurable'].get('summarise_depth')
    if recent_depth == 0:
        recent_depth = 1

    current_node_context = []
    if current_node_summary := current_node.summary:
        current_node_context.append(
            HumanMessage(f'前情提要（记忆总结）：{current_node_summary}')
        )

    context_nodes_trajectory = get_context_nodes_trajectory(tree_nodes, current_node_id)
    recent_context_nodes_trajectory = context_nodes_trajectory[-recent_depth:]
    for node in recent_context_nodes_trajectory:
        if node_action := node.action:
            current_node_context.append(
                AIMessage(
                    f'内部思考：{node_action.thought}\n需要调用的工具：{node_action.tool_name}\n需要调用的工具的参数：{node_action.tool_args}'
                )
            )
        if node_observation := node.observation:
            current_node_context.append(
                HumanMessage(f'工具运行情况：{node_observation}')
            )

    try:
        chain = ExpandGenerator(config['configurable'].get('llm')).get_extractor_chain()
        result = await chain.ainvoke(
            {
                'user_input_content': state.user_input_content,
                'summary': current_node_context,
                'input': '开始生成',
            },
            config,
        )

        if not (candidates := result.candidates):
            updated_node = current_node.model_copy()
            updated_node.is_completed = True
            return {
                'tree_nodes': {current_node_id: updated_node},
                'llm_call_count': state.llm_call_count + 1,
                'candidates': [],
            }

        return {
            'candidates': candidates,
            'llm_call_count': state.llm_call_count + 1,
        }
    except Exception:
        logger.error(f'<expander_node> 扩展器节点报错！！！\n{format_exc()}')
        return {'candidates': []}


async def process_expand_action(
    config: RunnableConfig,
    expand_action: ExpandAction,
    tool_manager: ToolManager,
    current_node_id: str,
    current_node_depth: int,
    current_node_summary: str | None,
) -> LATSTreeNode:
    """处理扩展行动"""

    tool = tool_manager.get_tool(tool_name := expand_action.tool_name)

    if not tool:
        logger.warning(f'<process_expand_action> 工具 {tool_name} 不存在！！！')
        observation = f'工具 {tool_name} 不存在！！！'
    elif not getattr(tool, 'is_safe', False):
        logger.warning(
            f'<process_expand_action> 工具 {tool_name} 不安全，已模拟运行，运行成功！！！'
        )
        observation = f'工具 {tool_name} 不安全，已模拟运行，运行成功！！！'
    else:
        try:
            tool_args = expand_action.tool_args or {}
            result = await tool.ainvoke(tool_args, config)
            observation = (
                result if isinstance(result, str) else dumps(result, ensure_ascii=False)
            )
        except Exception:
            logger.error(
                f'<process_expand_action> 处理扩展行动报错！！！\n{format_exc()}'
            )
            observation = f'工具 {tool_name} 执行失败。\n工具参数：{tool_args}\n'
    return LATSTreeNode(
        parent_id=current_node_id,
        action=expand_action,
        observation=observation,
        depth=current_node_depth + 1,
        summary=current_node_summary,
    )


async def executor_node(state: InferenceGraphState, config: RunnableConfig) -> dict:
    """执行器节点"""

    tool_manager = config['configurable'].get('tool_manager')
    current_node_id = state.current_node_id
    current_node = state.tree_nodes[current_node_id]

    results = await gather(
        *(
            process_expand_action(
                config,
                expand_action,
                tool_manager,
                current_node_id,
                current_node.depth,
                current_node.summary,
            )
            for expand_action in state.candidates
        ),
        return_exceptions=True,
    )

    new_nodes = {}
    for result in results:
        new_nodes[result.id] = result
    parent_node = current_node.model_copy()
    parent_node.children_ids.extend(list(new_nodes.keys()))
    return {'tree_nodes': new_nodes}
