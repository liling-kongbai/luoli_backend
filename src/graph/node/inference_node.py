from asyncio.tasks import gather
from json import dumps
from logging import getLogger
from math import log, sqrt
from traceback import format_exc

from langchain_core.runnables.config import RunnableConfig

from ..extractor import (
    EvaluateGenerator,
    ExpandGenerator,
    FinalExecutePlanGenerator,
    SummarizeGenerator,
)
from ..state import InferenceGraphState
from ..type import ExpandAction, LATSTreeNode, SelectionClassification

logger = getLogger(__name__)


def compute_uct_score(
    child_node_visit_count: int,
    child_node_score_count: float,
    parent_node_visit_count: int,
    exploration_c: float = 1.414,
) -> float:
    """计算 UCT 分数"""

    if child_node_visit_count == 0:
        return float('inf')

    exploitation_score = child_node_score_count / child_node_visit_count  # 利用分数
    exploration_score = exploration_c * sqrt(
        log(parent_node_visit_count) / child_node_visit_count
    )  # 探索分数
    return exploitation_score + exploration_score


def select_best_leaf_node(
    root_node_id: str, tree_nodes: dict[str, LATSTreeNode], exploration_c: float = 1.414
) -> str:
    """选择最佳叶子节点"""

    current_node_id = root_node_id
    while True:
        current_node = tree_nodes[current_node_id]
        if not current_node.child_ids:
            return current_node_id

        valid_child_nodes = []
        for child_id in current_node.child_ids:
            child_node = tree_nodes[child_id]
            if child_node and not child_node.is_pruned:
                valid_child_nodes.append(child_node)

        if not valid_child_nodes:
            return None

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


def inference_selector_node(state: InferenceGraphState, config: RunnableConfig) -> dict:
    """推理层选择器节点"""

    if state.iterate_count > config['configurable'].get('max_iterate_count', 10):
        logger.warning(
            '<inference_selector_node> 迭代计数超过最大迭代计数，停止推理！！！'
        )
        return {'current_node_id': SelectionClassification.Finalize.value}

    try:
        return {
            'current_node_id': select_best_leaf_node(
                state.root_node_id,
                state.tree_nodes,
                config['configurable'].get('exploration_c'),
            )
        }
    except Exception:
        logger.error(
            f'<inference_selector_node> 推理层选择器节点报错！！！\n{format_exc()}'
        )
        return {'current_node_id': SelectionClassification.Finalize.value}


def get_nodes_trajectory(
    tree_nodes: dict[str, LATSTreeNode], leaf_node_id: str, trajectory_depth: int
) -> list[LATSTreeNode]:
    """获取节点轨迹"""

    current_trajectory_depth = 0
    nodes_trajectory = []
    current_node_id = leaf_node_id
    while current_node_id and current_trajectory_depth <= trajectory_depth:
        current_node = tree_nodes[current_node_id]
        nodes_trajectory.append(current_node)
        current_node_id = current_node.parent_id
        current_trajectory_depth += 1
    return nodes_trajectory[::-1]


def get_nodes_context(trajectory: list[LATSTreeNode]) -> str:
    """获取节点上下文"""

    nodes_context = ''
    for node in trajectory:
        if node_action := node.action:
            nodes_context += f'思考：{node_action.thought}\n需要调用的工具：{node_action.tool_name}\n需要调用的工具的参数：{node_action.tool_args}\n'
        if node_observation := node.observation:
            nodes_context += f'工具运行的观察：{node_observation}\n'
    return nodes_context


async def inference_summarizer_node(
    state: InferenceGraphState, config: RunnableConfig
) -> dict:
    """推理层总结器节点"""

    current_node_id = state.current_node_id
    tree_nodes = state.tree_nodes
    current_node = tree_nodes[current_node_id]

    trajectory = get_nodes_trajectory(tree_nodes, current_node_id, 5)
    old_summary = trajectory[0].summary or ''

    chain = SummarizeGenerator(config['configurable'].get('llm')).get_extractor_chain()
    result = await chain.ainvoke(
        {
            'old_summary': old_summary,
            'recent_nodes_context': get_nodes_context(trajectory),
            'input': '开始总结',
        },
        config,
    )
    updated_node = current_node.model_copy()
    updated_node.summary = result.summary
    updated_node.summary_generate_depth = current_node.depth
    return {
        'tree_nodes': {current_node_id: updated_node},
        'llm_call_count': state.llm_call_count + 1,
    }


async def inference_expander_node(
    state: InferenceGraphState, config: RunnableConfig
) -> dict:
    """推理层扩展器节点"""

    trajectory = get_nodes_trajectory(state.tree_nodes, state.current_node_id, 5)
    old_summary = trajectory[0].summary or ''

    chain = ExpandGenerator(config['configurable'].get('llm')).get_extractor_chain()
    result = await chain.ainvoke(
        {
            'user_input_content': state.user_input_content,
            'node_context': f'前情提要/上下文总结：{old_summary}\n'
            + get_nodes_context(trajectory),
            'input': '开始生成',
        },
        config,
    )
    return {
        'candidates': result.candidates,
        'llm_call_count': state.llm_call_count + 1,
    }


async def execute_expand_action(
    config: RunnableConfig, expand_action: ExpandAction
) -> LATSTreeNode:
    """执行扩展行动"""

    if tool_name := expand_action.tool_name:
        if tool := config['configurable'].get('tool_manager').get_tool(tool_name):
            if getattr(tool, 'is_safe', True):
                try:
                    tool_args = expand_action.tool_args or {}
                    result = await tool.tool.ainvoke(tool_args, config)
                    observation = (
                        result
                        if isinstance(result, str)
                        else dumps(result, ensure_ascii=False)
                    )
                except Exception:
                    observation = f'工具 {tool_name} 运行报错！！！\n工具参数：{tool_args}\n{format_exc()}'
                    logger.error(f'<execute_expand_action> {observation}')
            else:
                observation = f'工具 {tool_name} 不安全，已模拟运行，运行完成！！！'
                logger.warning(f'<execute_expand_action> {observation}')
        else:
            observation = f'工具 {tool_name} 不存在，无法继续运行！！！'
            logger.warning(f'<execute_expand_action> {observation}')
    else:
        observation = '扩展行动无工具名称，无法继续运行！！！'
        logger.warning(f'<execute_expand_action> {observation}')
    return LATSTreeNode(action=expand_action, observation=observation)


async def inference_executor_node(
    state: InferenceGraphState, config: RunnableConfig
) -> dict:
    """推理层执行器节点"""

    current_node_id = state.current_node_id
    current_node = state.tree_nodes[current_node_id]
    if not (candidates := state.candidates):
        return {}

    results = await gather(
        *(execute_expand_action(config, expand_action) for expand_action in candidates),
        return_exceptions=True,
    )

    new_nodes = {}
    for result in results:
        if isinstance(result, Exception):
            error_node = LATSTreeNode(
                parent_id=current_node_id,
                depth=current_node.depth + 1,
                is_pruned=True,
                pruned_reason='运行扩展行动时报错，无法继续运行！！！',
            )
            new_nodes[error_node.id] = error_node
        result.parent_id = current_node_id
        result.depth = current_node.depth + 1
        result.summary = current_node.summary
        new_nodes[result.id] = result

    parent_node = current_node.model_copy()
    parent_node.child_ids.extend(list[str](new_nodes.keys()))
    new_nodes[current_node_id] = parent_node
    return {'tree_nodes': new_nodes}


async def evaluate_leaf_node(
    config: RunnableConfig,
    user_input_content: str,
    tree_nodes: dict[str, LATSTreeNode],
    leaf_node: LATSTreeNode,
) -> LATSTreeNode:
    """评估叶子节点"""

    chain = EvaluateGenerator(config['configurable'].get('llm')).get_extractor_chain()
    result = await chain.ainvoke(
        {
            'user_input_content': user_input_content,
            'nodes_context': get_nodes_context(
                get_nodes_trajectory(tree_nodes, leaf_node.id, leaf_node.depth)
            ),
            'input': '开始评估',
        },
        config,
    )
    new_node = leaf_node.model_copy()
    if result.score == 0 or result.is_pruned:
        new_node.is_pruned = True
        new_node.pruned_reason = result.analysis
    new_node.score_count = result.score
    new_node.is_completed = result.is_completed
    return new_node


async def inference_evaluator_node(
    state: InferenceGraphState, config: RunnableConfig
) -> dict:
    """推理层评估器节点"""

    tree_nodes = state.tree_nodes

    nodes_to_evaluate = []
    for child_id in tree_nodes[state.current_node_id].child_ids:
        if child_node := tree_nodes.get(child_id):
            if not child_node.is_pruned and child_node.visit_count == 0:
                nodes_to_evaluate.append(child_node)

    if not nodes_to_evaluate:
        return {}

    results = await gather(
        *(
            evaluate_leaf_node(config, state.user_input_content, tree_nodes, node)
            for node in nodes_to_evaluate
        ),
        return_exceptions=True,
    )

    new_nodes = {}
    for node, result in zip(nodes_to_evaluate, results):
        if isinstance(result, Exception):
            new_node = node.model_copy()
            new_node.visit_count += 1
            new_node.score_count = 0
            new_node.is_pruned = True
            new_node.pruned_reason = '评估叶子节点时报错，无法继续运行！！！'
            new_nodes[new_node.id] = new_node
        else:
            new_nodes[result.id] = result
    return {
        'tree_nodes': new_nodes,
        'llm_call_count': state.llm_call_count + len(nodes_to_evaluate),
    }


def backpropagate(
    tree_nodes: dict[str, LATSTreeNode], node_id: str, node_score_count: float
) -> dict[str, LATSTreeNode]:
    """反向传播"""

    updates = {}
    current_node_id = node_id

    while current_node_id:
        if not (current_node := tree_nodes.get(current_node_id)):
            break

        new_node = current_node.model_copy()
        new_node.visit_count += 1
        new_node.score_count += node_score_count

        if new_node.child_ids:
            is_pruned = True

            for child_id in new_node.child_ids:
                child_node = updates.get(child_id) or tree_nodes.get(child_id)
                if child_node and not child_node.is_pruned:
                    is_pruned = False
                    break

            if is_pruned and not new_node.is_pruned:
                new_node.is_pruned = True
                new_node.pruned_reason + '反向传播过程中，发现子节点被剪枝！！！'

        updates[new_node.id] = new_node
        current_node_id = new_node.get('parent_id')
    return updates


def inference_backpropagator_node(
    state: InferenceGraphState, config: RunnableConfig
) -> dict:
    """推理层反向传播器节点"""

    tree_nodes = state.tree_nodes

    valid_child_nodes = []
    for child_id in tree_nodes[state.current_node_id].child_ids:
        if child_node := tree_nodes.get(child_id):
            if child_node.visit_count == 0:
                valid_child_nodes.append(child_node)

    if not valid_child_nodes:
        return {'iterate_count': state.iterate_count + 1}

    new_tree_nodes = tree_nodes.copy()
    updates = {}
    for valid_child_node in valid_child_nodes:
        updates.update(
            backpropagate(
                tree_nodes=new_tree_nodes,
                node_id=valid_child_node.id,
                node_score_count=valid_child_node.score_count,
            )
        )
        new_tree_nodes.update(updates)
    return {'tree_nodes': new_tree_nodes, 'iterate_count': state.iterate_count + 1}


async def inference_final_node(
    state: InferenceGraphState, config: RunnableConfig
) -> dict:
    """推理层最终节点"""

    tree_nodes = state.tree_nodes

    final_node = None
    completed_nodes = [node for node in tree_nodes.values() if node.is_completed]
    if completed_nodes:
        final_node = max(completed_nodes, key=lambda node: node.score_count)
    if not final_node:
        return {}

    final_trajectory_content = ''
    for i, node in enumerate[LATSTreeNode](
        get_nodes_trajectory(tree_nodes, final_node.id, final_node.depth)
    ):
        node_action = node.action
        final_trajectory_content += f'步骤{i + 1}：思考：{node_action.thought}\n需要调用的工具：{node_action.tool_name}\n需要调用的工具的参数：{node_action.tool_args}\n工具运行的观察{node.observation}\n'

    chain = FinalExecutePlanGenerator(
        config['configurable'].get('llm')
    ).get_extractor_chain()
    result = await chain.ainvoke(
        {
            'user_input_content': state.user_input_content,
            'final_trajectory_content': final_trajectory_content,
            'input': '开始生成',
        },
        config,
    )
    return {'final_execute_plan': result}
