from pydantic.main import BaseModel


class DynamicConfig(BaseModel):
    """动态配置"""

    user_name: str = '理灵'
    max_iterate_count: int = 10
    summarise_depth: int = 3
    exploration_c: float = 1.414


class ConfigManager:
    """配置管理器"""

    def __init__(self):
        self._dynamic_config = DynamicConfig()
