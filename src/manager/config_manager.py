from psycopg_pool.pool_async import AsyncConnectionPool
from pydantic.main import BaseModel


class DynamicConfig(BaseModel):
    """动态配置"""

    user_name: str = '理灵'
    max_iterate_count: int = 10
    summarise_depth: int = 3
    exploration_c: float = 1.414


class ConfigManager:
    """配置管理器"""

    def __init__(self, user_id: str):
        self._user_id: str | None = None
        self._dynamic_config = DynamicConfig()

    async def load(self, connection_pool: AsyncConnectionPool):
        """加载"""

        async with connection_pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    'SELECT * FROM dynamic_config_table WHERE user_id = %s',
                    (self._user_id,),
                )
                result = await cursor.fetchone()
                if result:
                    self._dynamic_config = DynamicConfig(**result)
        return self._dynamic_config
