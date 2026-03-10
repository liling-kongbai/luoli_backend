from functools import partial

from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.prebuilt.tool_node import tools_condition

from .condition import (
    backpropagation_condition,
    intent_classifier_condition,
    intent_classifier_node,
    introspection_classifier_condition,
    introspection_classifier_node,
    selector_condition,
)
from .node import (
    inference_graph_adapter_node,
    intuition_chat_node,
    routine_chat_node,
    routine_final_chat_node,
    routine_graph_adapter_node,
    tool_call_node,
)
from .state import InferenceGraphState, MainGraphState, RoutineGraphState
from .type import (
    BackpropagationClassification,
    IntentClassification,
    IntrospectionClassification,
    SelectionClassification,
)


async def create_routine_graph() -> CompiledStateGraph:
    """创建常规层图"""

    routine_graph_builder = StateGraph(RoutineGraphState)
    routine_graph_builder.add_node('routine_chat_node', routine_chat_node)
    routine_graph_builder.add_node('tool_call_node', tool_call_node)
    routine_graph_builder.add_node(
        'introspection_classifier_node', introspection_classifier_node
    )
    routine_graph_builder.add_node('routine_final_chat_node', routine_final_chat_node)

    routine_graph_builder.add_edge(START, 'routine_chat_node')
    routine_graph_builder.add_conditional_edges(
        'routine_chat_node',
        tools_condition,
        {'tools': 'tool_call_node', '__end__': 'introspection_classifier_node'},
    )
    routine_graph_builder.add_edge('tool_call_node', 'routine_chat_node')
    routine_graph_builder.add_conditional_edges(
        'introspection_classifier_node',
        introspection_classifier_condition,
        {
            IntrospectionClassification.Introspect.value: 'routine_chat_node',
            IntrospectionClassification.Finalise.value: 'routine_final_chat_node',
        },
    )
    routine_graph_builder.add_edge('routine_final_chat_node', END)
    return routine_graph_builder.compile()


async def create_inference_graph() -> CompiledStateGraph:
    """创建推理层图"""

    inference_graph_builder = StateGraph(InferenceGraphState)
    inference_graph_builder.add_edge(START, 'selector_node')
    inference_graph_builder.add_conditional_edges(
        'selector_node',
        selector_condition,
        {
            SelectionClassification.Expand.value: 'expander_node',
            SelectionClassification.Summarise.value: 'summariser_node',
            SelectionClassification.Finalise.value: 'finaliser_node',
        },
    )
    inference_graph_builder.add_edge('summariser_node', 'expander_node')
    inference_graph_builder.add_edge('expander_node', 'executor_node')
    inference_graph_builder.add_edge('executor_node', 'evaluator_node')
    inference_graph_builder.add_edge('evaluator_node', 'backpropagation_node')
    inference_graph_builder.add_conditional_edges(
        'backpropagation_node',
        backpropagation_condition,
        {
            BackpropagationClassification.Select.value: 'selector_node',
            BackpropagationClassification.Finalise.value: 'finaliser_node',
        },
    )
    inference_graph_builder.add_edge('finaliser_node', END)
    return inference_graph_builder.compile()


async def create_main_graph(
    routine_graph: CompiledStateGraph, inference_graph: CompiledStateGraph
) -> CompiledStateGraph:
    """创建主图"""

    mian_graph_builder = StateGraph(MainGraphState)
    mian_graph_builder.add_node('intent_classifier_node', intent_classifier_node)
    mian_graph_builder.add_node('intuition_chat_node', intuition_chat_node)
    mian_graph_builder.add_node(
        'routine_graph_adapter_node',
        partial(routine_graph_adapter_node, routine_graph=routine_graph),
    )
    mian_graph_builder.add_node(
        'inference_graph_adapter_node',
        partial(inference_graph_adapter_node, inference_graph=inference_graph),
    )

    mian_graph_builder.add_edge(START, 'intent_classifier_node')
    mian_graph_builder.add_conditional_edges(
        'intent_classifier_node',
        intent_classifier_condition,
        {
            IntentClassification.IntuitionLayer.value: 'intuition_chat_node',
            IntentClassification.RoutineLayer.value: 'routine_graph_adapter_node',
            IntentClassification.InferenceLayer.value: 'inference_graph_adapter_node',
        },
    )
    mian_graph_builder.add_edge('intuition_chat_node', END)
    mian_graph_builder.add_edge('routine_graph_adapter_node', END)
    mian_graph_builder.add_edge(
        'inference_graph_adapter_node', 'routine_graph_adapter_node'
    )
    return mian_graph_builder.compile()
