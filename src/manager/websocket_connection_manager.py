from asyncio.tasks import gather
from logging import getLogger
from traceback import format_exc

from fastapi import WebSocket

logger = getLogger(__name__)


class WebSocketConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self._connections: dict[str, set[WebSocket]] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        """连接"""

        await websocket.accept()

        if user_id not in self._connections:
            self._connections[user_id] = set[WebSocket]()

        self._connections[user_id].add(websocket)
        logger.info(
            f'<connect> 用户 {user_id} 连接 WebSocket 成功，当前连接数：{len(self._connections[user_id])}'
        )

    def disconnect(self, user_id: str, websocket: WebSocket):
        """断开连接"""

        if user_id in self._connections:
            if websocket in self._connections[user_id]:
                self._connections[user_id].remove(websocket)
                logger.info(
                    f'<disconnect> 用户 {user_id} 断开 WebSocket 连接，当前连接数：{len(self._connections[user_id])}'
                )

            if not self._connections[user_id]:
                del self._connections[user_id]
                logger.info(f'<disconnect> 用户 {user_id} 已无连接，已清理用户')

    async def _safe_send_message(
        self, user_id: str, websocket: WebSocket, message: dict
    ):
        """安全发送消息"""

        try:
            await websocket.send_json(message)
        except Exception:
            logger.error(f'<_safe_send_message> 安全发送消息报错！！！\n{format_exc()}')
            self.disconnect(user_id, websocket)
            raise

    async def send_message(
        self,
        user_id: str,
        message_type: str,
        message_payload: any,
        message_thread_id: str | None = None,
    ):
        """发送消息"""

        message = {
            'luoli_backend_message_type': message_type,
            'luoli_backend_message_payload': message_payload,
            'luoli_backend_message_thread_id': message_thread_id,
        }

        tasks = [
            self._safe_send_message(user_id, websocket, message)
            for websocket in self._connections[user_id]
        ]
        await gather(*tasks, return_exceptions=True)

    async def broadcast(self, message_type: str, message_payload: any):
        """广播"""

        for user_id in self._connections:
            await self.send_message(
                user_id,
                {
                    'luoli_backend_message_type': message_type,
                    'luoli_backend_message_payload': message_payload,
                },
            )
