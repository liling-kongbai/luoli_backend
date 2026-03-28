from logging import getLogger
from traceback import format_exc

from langchain_core.runnables.config import RunnableConfig

from ..extractor import IntentClassifier, IntrospectionClassifier
from ..state import InferenceGraphState, MainGraphState, RoutineGraphState
from ..type import (
    BackpropagationClassification,
    IntentClassification,
    IntrospectionClassification,
    SelectionClassification,
)

logger = getLogger(__name__)


async def intent_classifier_node(state: MainGraphState, config: RunnableConfig) -> dict:
    """意图分类器节点"""

    try:
        chain = IntentClassifier(
            config['configurable'].get('llm')
        ).get_extractor_chain()
        result = await chain.ainvoke(
            {'messages': state.messages[-10:], 'input': '开始分类'}, config
        )
        return {'intent': result.intent.value}
    except Exception:
        logger.error(
            f'<intent_classifier_node> 意图分类器节点报错！！！\n{format_exc()}'
        )
        return {'intent': IntentClassification.RoutineLayer.value}


def intent_classifier_condition(state: MainGraphState) -> str:
    """意图分类器条件"""

    return state.intent


async def introspect_classifier_node(
    state: RoutineGraphState, config: RunnableConfig
) -> dict:
    """反思分类器节点"""

    introspect_count = state.introspect_count
    if introspect_count >= 3:
        logger.warning(
            '<introspect_classifier_node> 反思次数超过 3 次，直接返回最终！！！'
        )
        return {'introspection': IntrospectionClassification.Finalize.value}

    try:
        chain = IntrospectionClassifier(
            config['configurable'].get('llm')
        ).get_extractor_chain()
        result = await chain.ainvoke(
            {
                'messages': state.messages,
                'response_draft': state.response_draft_content,
                'input': f'本次用户的消息/问题：{state.user_input_content}',
            },
            config,
        )
        return {
            'introspect_count': introspect_count + 1,
            'introspection': result.introspection.value,
            'introspect_reason': result.reason,
        }
    except Exception:
        logger.error(
            f'<introspect_classifier_node> 反思分类器节点报错！！！\n{format_exc()}'
        )
        return {'introspection': IntrospectionClassification.Finalize.value}


def introspect_classifier_condition(state: RoutineGraphState) -> str:
    """反思分类器条件"""

    return state.introspection


def selector_condition(state: InferenceGraphState, config: RunnableConfig) -> str:
    """选择器条件"""

    current_node_id = state.current_node_id
    if not current_node_id or current_node_id == SelectionClassification.Finalize.value:
        logger.warning(
            '<selector_condition> 当前节点 ID 不存在或为最终，直接返回最终！！！'
        )
        return SelectionClassification.Finalize.value

    current_node = state.tree_nodes[current_node_id]
    if current_node_id == state.root_node_id:
        if state.iterate_count > 0 and not current_node.child_ids:
            logger.warning(
                '<selector_condition> 迭代计数大于 0 且根节点没有子节点，直接返回最终！！！'
            )
            return SelectionClassification.Finalize.value

    current_node_depth = current_node.depth
    if current_node_depth > 0:
        if (
            current_node_depth % config['configurable'].get('summarize_depth', 5) == 0
            and current_node_depth > current_node.summary_generate_depth
        ):
            return SelectionClassification.Summarize.value
    return SelectionClassification.Expand.value


def route_from_backprop(state, config: RunnableConfig) -> str:
    tree_nodes = state.tree_nodes

    # 1. 检查“提前终止”信号 (Early Stopping)
    # 只要整棵树里出现了任何一个 is_completed=True 的节点，立刻收工！
    for node in tree_nodes.values():
        if getattr(node, 'is_completed', False):
            return 'finaliser_node'

    # 2. 检查“全军覆没”信号
    # 如果根节点的所有子分支都被标记为 is_pruned，也没必要再搜了
    root = tree_nodes.get(state.root_id)
    if root and root.child_ids:
        all_root_children_dead = all(
            tree_nodes[cid].is_pruned for cid in root.child_ids
        )
        if all_root_children_dead:
            return 'finaliser_node'

    # 3. 继续循环：回到 Selector 开始下一轮 UCT 选择
    return 'selector_node'
