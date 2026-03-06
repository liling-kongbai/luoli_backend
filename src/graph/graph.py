from functools import partial

from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.prebuilt.tool_node import tools_condition

from .condition import (
    intent_classifier_condition,
    intent_classifier_node,
    introspect_classifier_condition,
    introspect_classifier_node,
)
from .node import (
    intuition_chat_node,
    routine_chat_node,
    routine_final_chat_node,
    routine_graph_adapter_node,
    tools_node,
)
from .state import MainGraphState, RoutineGraphState
from .type import IntentClassification, IntrospectionClassification


async def create_routine_graph() -> CompiledStateGraph:
    """创建常规层图"""

    routine_graph_builder = StateGraph(RoutineGraphState)

    routine_graph_builder.add_node('routine_chat_node', routine_chat_node)
    routine_graph_builder.add_node('tool_node', tools_node)
    routine_graph_builder.add_node(
        'introspect_classifier_node', introspect_classifier_node
    )
    routine_graph_builder.add_node('routine_final_chat_node', routine_final_chat_node)

    routine_graph_builder.add_edge(START, 'routine_chat_node')
    routine_graph_builder.add_conditional_edges(
        'routine_chat_node',
        tools_condition,
        {'tools': 'tool_node', '__end__': 'introspect_classifier_node'},
    )
    routine_graph_builder.add_edge('tool_node', 'routine_chat_node')
    routine_graph_builder.add_conditional_edges(
        'introspect_classifier_node',
        introspect_classifier_condition,
        {
            IntrospectionClassification.IntrospectLayer.value: 'routine_chat_node',
            IntrospectionClassification.FinalChatLayer.value: 'routine_final_chat_node',
        },
    )
    routine_graph_builder.add_edge('routine_final_chat_node', END)
    return routine_graph_builder.compile()


async def create_main_graph(routine_graph: CompiledStateGraph) -> CompiledStateGraph:
    """创建主图"""

    builder = StateGraph(MainGraphState)

    builder.add_node('intent_classifier_node', intent_classifier_node)
    builder.add_node('intuition_chat_node', intuition_chat_node)
    builder.add_node(
        'routine_graph_adapter_node',
        partial(routine_graph_adapter_node, routine_graph=routine_graph),
    )

    builder.add_edge(START, 'intent_classifier_node')
    builder.add_conditional_edges(
        'intent_classifier_node',
        intent_classifier_condition,
        {
            IntentClassification.IntuitionLayer.value: 'intuition_chat_node',
            IntentClassification.RoutineLayer.value: 'routine_graph_adapter_node',
            IntentClassification.InferenceLayer.value: '',
        },
    )
    builder.add_edge('intuition_chat_node', END)
    builder.add_edge('routine_graph_adapter_node', END)
    return builder.compile()
