from asyncio.tasks import gather
from json import dumps
from logging import getLogger
from math import log, sqrt
from traceback import format_exc

from langchain_core.runnables.config import RunnableConfig

from ..extractor import EvaluateGenerator, ExpandGenerator, FinalPlanGenerator
from ..state import InferenceGraphState
from ..type import ExecutionPlan, ExpandAction, LATSTreeNode, SelectionClassification

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
    tree_nodes: dict[str, LATSTreeNode], leaf_node_id: str, context_depth: int
) -> list[LATSTreeNode]:
    """获取上下文节点轨迹"""

    current_context_depth = 0
    context_nodes_trajectory = []
    current_node_id = leaf_node_id
    while current_node_id and current_context_depth <= context_depth:
        current_node = tree_nodes[current_node_id]
        context_nodes_trajectory.append(current_node)
        current_node_id = current_node.parent_id
        current_context_depth += 1
    return context_nodes_trajectory[::-1]


async def expander_node(state: InferenceGraphState, config: RunnableConfig) -> dict:
    """扩展器节点"""

    current_node_id = state.current_node_id
    tree_nodes = state.tree_nodes
    current_node = tree_nodes[current_node_id]

    current_node_context = ''

    recent_depth = current_node.depth % config['configurable'].get('summarise_depth')
    if current_node_summary := current_node.summary:
        current_node_context += f'前情提要/记忆总结：{current_node_summary}\n'

    if recent_depth != 0 or not current_node_summary:
        context_nodes_trajectory = get_context_nodes_trajectory(
            tree_nodes, current_node_id, recent_depth
        )
        for node in context_nodes_trajectory:
            if node_action := node.action:
                current_node_context += f'内部思考：{node_action.thought}\n需要调用的工具：{node_action.tool_name}\n需要调用的工具的参数：{node_action.tool_args}\n'
            if node_observation := node.observation:
                current_node_context += f'工具运行情况：{node_observation}\n'

    try:
        chain = ExpandGenerator(config['configurable'].get('llm')).get_extractor_chain()
        result = await chain.ainvoke(
            {
                'user_input_content': state.user_input_content,
                'current_node_context': current_node_context,
                'input': '开始生成',
            },
            config,
        )
        return {
            'llm_call_count': state.llm_call_count + 1,
            'candidates': result.candidates or [],
        }
    except Exception:
        logger.error(f'<expander_node> 扩展器节点报错！！！\n{format_exc()}')
        return {'candidates': []}


async def process_expand_action(
    config: RunnableConfig,
    expand_action: ExpandAction,
    current_node_id: str,
    current_node_depth: int,
    current_node_summary: str | None,
) -> LATSTreeNode:
    """处理扩展行动"""

    if tool_name := expand_action.tool_name:
        if tool := config['configurable'].get('tool_manager').get_tool(tool_name):
            if getattr(tool, 'is_safe', False):
                logger.warning(
                    f'<process_expand_action> 工具 {tool_name} 不安全，已模拟运行，运行完成！！！'
                )
                observation = f'工具 {tool_name} 不安全，已模拟运行，运行完成！！！'
            else:
                try:
                    tool_args = expand_action.tool_args or {}
                    result = await tool.tool.ainvoke(tool_args, config)
                    observation = (
                        result
                        if isinstance(result, str)
                        else dumps(result, ensure_ascii=False)
                    )
                except Exception:
                    logger.error(
                        f'<process_expand_action> 处理扩展行动报错！！！\n{format_exc()}'
                    )
                    observation = f'工具 {tool_name} 执行失败。\n工具参数：{tool_args}\n{format_exc()}'
        else:
            logger.warning(f'<process_expand_action> 工具 {tool_name} 不存在！！！')
            observation = f'工具 {tool_name} 不存在！！！'
    return LATSTreeNode(
        parent_id=current_node_id,
        depth=current_node_depth + 1,
        summary=current_node_summary,
        action=expand_action,
        observation=observation,
    )


async def executor_node(state: InferenceGraphState, config: RunnableConfig) -> dict:
    """执行器节点"""

    current_node_id = state.current_node_id
    current_node = state.tree_nodes[current_node_id]

    results = await gather(
        *(
            process_expand_action(
                config,
                expand_action,
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
        if not isinstance(result, Exception):
            new_nodes[result.id] = result
    parent_node = current_node.model_copy()
    parent_node.children_ids.extend(list(new_nodes.keys()))
    new_nodes[current_node_id] = parent_node
    return {'tree_nodes': new_nodes}


async def evaluate_tree_node(
    config: RunnableConfig, node: LATSTreeNode, user_input_content: str
) -> LATSTreeNode:
    """评估树节点"""

    evaluate_message = f'动作：{node.action.thought}\n工具：{node.action.tool_name}\n工具参数：{node.action.tool_args}\n观察：{node.observation}'
    try:
        chain = EvaluateGenerator(
            config['configurable'].get('llm')
        ).get_extractor_chain()
        result = await chain.ainvoke(
            {
                'user_input_content': user_input_content,
                'input': evaluate_message,
            },
            config,
        )
        new_node = node.model_copy()
        if result.is_pruned or result.score == 0:
            new_node.is_pruned = True
            new_node.pruned_reason = result.analysis
        new_node.score_count = result.score
        new_node.is_completed = result.is_completed
        return new_node
    except Exception:
        logger.error(f'<evaluate_tree_node> 评估树节点报错！！！\n{format_exc()}')
        new_node = node.model_copy()
        new_node.is_pruned = True
        new_node.pruned_reason = '评估树节点报错'
        return new_node


async def evaluator_node(state: InferenceGraphState, config: RunnableConfig) -> dict:
    """评估器节点"""

    parent_node_id = state.current_node_id
    parent_node = state.tree_nodes[parent_node_id]

    nodes_to_evaluate = []
    for child_id in parent_node.child_ids:
        child_node = state.tree_nodes[child_id]
        if child_node and child_node.visit_count == 0:
            nodes_to_evaluate.append(child_node)

    if not nodes_to_evaluate:
        return {}

    results = await gather(
        *(
            evaluate_tree_node(config, node, state.user_input_content)
            for node in nodes_to_evaluate
        ),
        return_exceptions=True,
    )
    return {
        'tree_nodes': {node.id: node for node in results},
        'llm_call_count': state.llm_call_count + len(nodes_to_evaluate),
    }


def backpropagate(
    tree_nodes: dict[str, LATSTreeNode], leaf_node_id: str
) -> dict[str, LATSTreeNode]:
    """反向传播"""

    updates = {}
    current_node_id = leaf_node_id

    while current_node_id:
        current_node = tree_nodes.get(current_node_id)
        if not current_node:
            break

        new_node = current_node.model_copy()
        new_node.visit_count += 1
        new_node.score_count += current_node.score_count

        is_pruned = True
        if new_node.child_ids:
            for child_id in new_node.child_ids:
                child_node = tree_nodes.get(child_id)
                if not child_node or child_node.is_pruned:
                    is_pruned = False
                    break
            if is_pruned:
                new_node.is_pruned = True
                new_node.pruned_reason = '反向传播过程中，发现子节点被剪枝！！！'

        updates[new_node.id] = new_node
        current_node_id = new_node.parent_id
        return updates


def backpropagator_node(state: InferenceGraphState, config: RunnableConfig) -> dict:
    """反向传播器节点"""

    parent_id = state.current_node_id
    parent_node = state.tree_nodes[parent_id]

    leaf_nodes = []
    for child_id in parent_node.child_ids:
        child_node = state.tree_nodes[child_id]
        if child_node and child_node.visit_count == 0:
            leaf_nodes.append(child_node)

    updates = {}
    for leaf_node in leaf_nodes:
        updates = backpropagate(
            tree_nodes=state.tree_nodes,
            leaf_node_id=leaf_node.id,
        )
        updates.update(updates)
    return {'tree_nodes': updates, 'iterate_count': state.iterate_count + 1}


async def finaliser_node(state: InferenceGraphState, config: RunnableConfig) -> dict:
    """最终器节点"""

    tree_nodes = state.tree_nodes
    root_node_id = state.root_id

    best_node = None
    best_score = -float('inf')

    for node in tree_nodes.values():
        if node.is_pruned:
            continue

        if node.is_completed:
            best_node = node
            break

        score = node.score_count / node.visit_count
        if node.visit_count > 0 and score > best_score:
            best_score = score
            best_node = node

    if not best_node:
        return tree_nodes.get(root_node_id)

    trajectory = get_context_nodes_trajectory(tree_nodes, best_node.id)
    trajectory_summary = '\n'.join(
        f'步骤{i + 1}：内部思考：{node.action.thought}\n需要调用的工具：{node.action.tool_name}\n需要调用的工具的参数：{node.action.tool_args}\n工具运行情况：{node.observation}'
        for i, node in enumerate(trajectory)
    )

    try:
        chain = FinalPlanGenerator(
            config['configurable'].get('llm')
        ).get_extractor_chain()
        result = await chain.ainvoke(
            {
                'user_input_content': state.user_input_content,
                'trajectory': trajectory_summary,
                'input': '开始生成执行计划',
            },
            config,
        )
        return {'final_plan': result.final_plan}
    except Exception:
        logger.error(f'<finaliser_node> 最终器节点报错！！！\n{format_exc()}')
        return {
            'final_plan': ExecutionPlan(
                original_goal=state.user_input_content, steps=[]
            )
        }
