from logging import getLogger
from traceback import format_exc

from langchain_core.runnables.config import RunnableConfig

from .manager import GraphManager, WebSocketConnectionManager

logger = getLogger(__name__)


class Agent:
    def __init__(self):
        # 管理器相关
        self._websocket_connection_manager: WebSocketConnectionManager | None = None
        self._graph_manager: GraphManager | None = None

    async def _process_chat(
        self,
        input: str,
        config: RunnableConfig,
        thread_id: str,
        user_id: str,
    ):
        """处理对话"""

        try:
            events = await self._graph_manager.stream_chat(input, config)
            async for event in events:
                type = event['luoli_backend_type']
                payload = event['luoli_backend_payload']
                match type:
                    case 'ai_message_chunk':
                        await self._websocket_connection_manager.send_message(
                            user_id, 'ai_message_chunk', payload, thread_id
                        )
                    case 'graph_event':
                        await self._websocket_connection_manager.send_message(
                            user_id, 'graph_event', payload, thread_id
                        )
                    case _:
                        logger.warning(
                            f'<_process_chat> 处理对话不支持的事件类型：{type}'
                        )
                        await self._websocket_connection_manager.send_message(
                            user_id,
                            'agent_event',
                            f'处理对话不支持的事件类型：{type}',
                            thread_id,
                        )
        except Exception:
            logger.error(f'<_process_chat> 处理对话报错！！！\n{format_exc()}')
            raise
