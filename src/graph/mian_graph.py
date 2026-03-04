from langgraph.graph import END, START, StateGraph

from .condition import intent_classifier_condition
from .node import intuition_chat_node
from .type import IntentClassification


async def create_main_graph_builder():
    """创建主图构建器"""

    builder = StateGraph()

    builder.add_node('intuition_chat_node', intuition_chat_node)

    builder.add_conditional_edges(
        START,
        intent_classifier_condition,
        {
            IntentClassification.IntuitionLayer.value: 'intuition_chat_node',
            IntentClassification.RoutineLayer.value: '',
            IntentClassification.InferenceLayer.value: '',
        },
    )

    builder.add_edge('intuition_chat_node', END)
    return builder
