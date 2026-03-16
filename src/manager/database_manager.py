from logging import getLogger
from traceback import format_exc

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.connection_async import AsyncConnection
from psycopg_pool.pool_async import AsyncConnectionPool

logger = getLogger(__name__)


class DatabaseManager:
    """数据库管理器"""

    def __init__(self):
        self._postgresql_connection_string: str | None = (
            'postgresql://postgres:root@localhost:5432/test'
        )
        self._postgresql_connection_pool: AsyncConnectionPool | None = None
        self._async_postgresql_saver: AsyncPostgresSaver | None = None

    async def init_postgresql(self):
        """初始化 PostgreSQL 数据库相关内容"""

        try:
            async with await AsyncConnection.connect(
                self._postgresql_connection_string, autocommit=True
            ) as connection:
                await AsyncPostgresSaver(connection).setup()
            logger.info('<init_postgresql> 初始化 PostgreSQL 数据库准备动作完成')

            self._postgresql_connection_pool = AsyncConnectionPool(
                self._postgresql_connection_string, min_size=3, max_size=5, open=False
            )
            await self._postgresql_connection_pool.open()
            logger.info('<init_postgresql> 初始化 PostgreSQL 数据库连接池完成')

            self._async_postgresql_saver = AsyncPostgresSaver(
                self._postgresql_connection_pool
            )
            logger.info('<init_postgresql> 初始化 PostgreSQL 数据库检查点保存器完成')
        except Exception:
            logger.error(
                f'<init_postgresql> 初始化 PostgreSQL 数据库相关内容报错！！！\n{format_exc()}'
            )
            raise
