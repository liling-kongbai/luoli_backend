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
            '<introspect_classifier_node> 反思次数超过 3 次，直接返回最终响应层！！！'
        )
        return {'introspection': IntrospectionClassification.Finalize.value}

    try:
        chain = IntrospectionClassifier(
            config['configurable'].get('llm')
        ).get_extractor_chain()
        result = await chain.ainvoke(
            {
                'messages': state.messages,
                'response_draft': state.response_draft.content
                if state.response_draft
                else '暂时没有响应草稿',
                'input': f'本次用户的消息/问题：{state.user_input_content}',
            },
            config,
        )

        if (
            result.introspection.value
            == IntrospectionClassification.FinalChatLayer.value
        ):
            return {'introspection': result.introspection.value}
        else:
            return {
                'introspect_count': introspect_count + 1,
                'introspection': result.introspection.value,
                'introspect_reason': result.reason,
            }
    except Exception:
        logger.error(
            f'<introspect_classifier_node> 反思分类器节点报错！！！\n{format_exc()}'
        )
        return {'introspection': IntrospectionClassification.FinalChatLayer.value}


def introspect_classifier_condition(state: RoutineGraphState) -> str:
    """反思分类器条件"""

    return state.introspection


def inference_selector_condition(state: InferenceGraphState) -> str:
    """推理选择器条件"""

    pass


def route_from_selector(state: InferenceGraphState, config: RunnableConfig) -> str:

    current_node_id = state.current_node_id

    # 1. 熔断/结算：如果 Selector 返回 None (说明达到迭代上限)
    # 或者所有叶子节点都被剪枝了（无路可走）
    if not current_node_id or current_node_id == SelectionClassification.Finalize.value:
        return 'finalize_node'

    tree_nodes = state.tree_nodes
    current_node = tree_nodes[current_node_id]

    # 2. 判定是否触发“逢五抽一”总结
    summarize_depth = config['configurable'].get('summarize_depth', 5)

    # 条件：深度是 5 的倍数，且该节点目前还没有摘要
    # 注意：depth=0 是根节点，通常不需要总结
    if (
        current_node.depth > 0
        and current_node.depth % summarize_depth == 0
        and not getattr(current_node, 'summary', None)
    ):
        return 'summary_node'

    # 3. 正常情况：前往扩展节点
    return 'expander_node'


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
