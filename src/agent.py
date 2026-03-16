from asyncio.tasks import create_task
from json import loads
from logging import getLogger
from traceback import format_exc

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools.base import BaseTool

from .manager import GraphManager, WebSocketConnectionManager

logger = getLogger(__name__)


class Agent:
    def __init__(self):
        self._user_id: str = 'liling'

        # 管理器相关
        self._websocket_connection_manager: WebSocketConnectionManager | None = None
        self._graph_manager: GraphManager | None = None

        self._llm: BaseChatModel | None = None
        self._tools: list[BaseTool] | None = None

        self._user_name: str = '理灵'
        self._max_iterate_count: int = 10
        self._summarise_depth: int = 3
        self._exploration_c: float = 1.414

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

    async def process_request(self, request: str):
        """处理请求"""

        data = loads(request)
        action = data.get('action')
        match action:
            case 'chat':
                create_task(
                    self._process_chat(
                        data.get('input'),
                        RunnableConfig(
                            configurable={
                                'llm': self._llm,
                                'tools': self._tools,
                                'user_name': self._user_name,
                                'max_iterate_count': self._max_iterate_count,
                                'summarise_depth': self._summarise_depth,
                                'exploration_c': self._exploration_c,
                            },
                        ),
                        data.get('thread_id'),
                        self._user_id,
                    )
                )
            case _:
                logger.warning(f'<process_request> 处理请求不支持的动作：{action}')
