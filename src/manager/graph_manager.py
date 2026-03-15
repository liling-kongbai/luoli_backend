from logging import getLogger
from traceback import format_exc
from typing import Any, AsyncGenerator

from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.human import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from ..graph import create_inference_graph, create_main_graph, create_routine_graph

logger = getLogger(__name__)


class GraphManager:
    """图管理器"""

    def __init__(self, checkpoint_saver: BaseCheckpointSaver):
        self._checkpoint_saver: BaseCheckpointSaver | None = checkpoint_saver
        self._graph: CompiledStateGraph | None = None

    def clean(self):
        """清理"""

        self._graph = None
        self._checkpoint_saver = None
        logger.info('<clean> 图管理器已清理')

    def compile_graph(self):
        """编译图"""

        if self._graph:
            logger.warning(
                '<compile_graph> 图已存在，请勿重复编译，请检查代码逻辑！！！'
            )
            return

        if not self._checkpoint_saver:
            logger.warning(
                '<compile_graph> 未编译图，检查点保存器不存在，请检查代码逻辑！！！'
            )
            return

        try:
            logger.info('<compile_graph> 开始编译图')
            self._graph = create_main_graph(
                routine_graph=create_routine_graph(),
                inference_graph=create_inference_graph(),
                checkpoint_saver=self._checkpoint_saver,
            )
            logger.info('<compile_graph> 编译图完成')
        except Exception:
            logger.error(f'<compile_graph> 编译图失败！！！\n{format_exc()}')
            raise

    async def stream_chat(
        self,
        input: str,
        config: RunnableConfig,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """流式对话"""

        async for event in self._graph.astream_events(
            {'messages': [HumanMessage(input)]}, config
        ):
            event_type = event.get('event')
            node_name = event.get('metadata', {}).get('langgraph_node')

            match event_type:
                case 'on_chat_model_stream':
                    chunk = event.get('data')['chunk']
                    if isinstance(chunk, AIMessageChunk):
                        yield {
                            'luoli_backend_type': 'ai_message_chunk',
                            'luoli_backend_payload': chunk.content,
                        }
                case 'on_chain_start':
                    yield {
                        'luoli_backend_type': 'graph_event',
                        'luoli_backend_payload': f'{node_name} 开始运行',
                    }
                case 'on_chain_end':
                    yield {
                        'luoli_backend_type': 'graph_event',
                        'luoli_backend_payload': f'{node_name} 结束运行，运行结果如下：\n{event.get("data", {}).get("output")}',
                    }
                case _:
                    logger.warning(
                        f'<stream_chat> 图管理器不支持的事件类型：{event_type}，产生于节点：{node_name}'
                    )
                    yield {
                        'luoli_backend_type': 'graph_event',
                        'luoli_backend_payload': f'图管理器不支持的事件类型：{event_type}，产生于节点：{node_name}',
                    }
