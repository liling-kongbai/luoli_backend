from logging import getLogger
from math import log, sqrt
from traceback import format_exc

from langchain_core.runnables.config import RunnableConfig

from ..state import InferenceGraphState
from ..type import LATSTreeNode, SelectionClassification

logger = getLogger(__name__)


def compute_uct_score(
    child_visit_count: int,
    child_score_count: int,
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
