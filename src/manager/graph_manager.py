from logging import getLogger
from traceback import format_exc
from typing import AsyncGenerator

from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.human import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from ..graph import create_main_graph_builder

logger = getLogger(__name__)


class GraphManager:
    """图管理器"""

    def __init__(self, checkpoint_saver: BaseCheckpointSaver):
        self._checkpoint_saver: BaseCheckpointSaver | None = checkpoint_saver
        self._graph: CompiledStateGraph | None = None

    async def compile_graph(self):
        """编译图"""

        if self._graph:
            logger.warning(
                '<compile_graph> 图已存在，请勿重复编译，请检查代码逻辑！！！'
            )
            return

        try:
            logger.info('<compile_graph> 开始编译图')
            builder = await create_main_graph_builder()
            self._graph = builder.compile(self._checkpoint_saver)
            logger.info('<compile_graph> 图编译成功')
        except Exception:
            logger.error(f'<compile_graph> 编译图失败！！！\n{format_exc()}')
            raise

    async def stream_chat(
        self,
        config: RunnableConfig,
        user_input_content: str,
    ) -> AsyncGenerator[dict, None]:
        """流式对话"""

        if not self._graph:
            logger.warning('<stream_chat> 图未编译！！！')
            await self.compile_graph()

        try:
            async for event in self._graph.astream_events(
                {'messages': [HumanMessage(content=user_input_content)]},
                config,
            ):
                event_type = event['event']

                if event_type == 'on_chat_model_stream':
                    chunk = event['data']['chunk']
                    if isinstance(chunk, AIMessageChunk):
                        yield {
                            'luoli_backend_type': 'ai_message_chunk',
                            'luoli_backend_payload': chunk.content,
                        }
        except Exception:
            logger.error(f'<stream_chat> 流式对话报错！！！\n{format_exc()}')
            raise
