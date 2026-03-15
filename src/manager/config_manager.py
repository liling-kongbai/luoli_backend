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
        self._user_id: str = user_id
        self._dynamic_config = DynamicConfig()

    async def load(self, connection_pool: AsyncConnectionPool) -> DynamicConfig:
        """加载"""

        async with connection_pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    'SELECT user_name, max_iterate_count, summarise_depth, exploration_c FROM config_table WHERE user_id = %s',
                    (self._user_id,),
                )
                result = await cursor.fetchone()
                if result:
                    self._dynamic_config = DynamicConfig(
                        user_name=result[0],
                        max_iterate_count=result[1],
                        summarise_depth=result[2],
                        exploration_c=result[3],
                    )
        return self._dynamic_config

    async def update(
        self, connection_pool: AsyncConnectionPool, dynamic_config: DynamicConfig
    ):
        """更新"""

        self._dynamic_config = dynamic_config
        async with connection_pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    'UPDATE config_table SET user_name = %s, exploration_c = %s, max_iterate_count = %s, summarise_depth = %s WHERE user_id = %s',
                    (
                        dynamic_config.user_name,
                        dynamic_config.exploration_c,
                        dynamic_config.max_iterate_count,
                        dynamic_config.summarise_depth,
                        self._user_id,
                    ),
                )
                await connection.commit()
