from logging import getLogger
from traceback import format_exc

from langchain_core.language_models.chat_models import BaseChatModel

from ..utils import connect_deepseek_llm, connect_ollama_llm

logger = getLogger(__name__)


class LLMManager:
    """LLM 管理器"""

    def __init__(self):
        self._current_llm_platform_name: str | None = None
        self._current_llm_model_name: str | None = None
        self._current_llm_tools: list = []
        self._current_llm: BaseChatModel | None = None
        self._current_llm_bind_tools = self._current_llm

        self._current_embedding_model = None

    def clean_llm(self):
        """清理 llm"""

        self._current_llm = None
        self._current_llm_bind_tools = None

    def llm_bind_tools(self):
        """LLM 绑定工具"""

        try:
            if self._current_llm_tools:
                self._current_llm_bind_tools = self._current_llm.bind_tools(
                    self._current_llm_tools
                )
            else:
                self._current_llm_bind_tools = self._current_llm
        except Exception:
            self._current_llm_bind_tools = self._current_llm
            logger.error(
                f'<llm_bind_tools> {self._current_llm_platform_name} 平台的 {self._current_llm_model_name} 绑定工具失败！！！\n{format_exc()}'
            )
            raise

    def connect_llm(
        self, platform: str, model: str, temperature: float | None = None, **kwargs
    ):
        """连接 LLM"""

        logger.info(f'<connect_llm> 连接 {platform} 平台的 {model}')

        try:
            match platform:
                case 'ollama':
                    self._current_llm = connect_ollama_llm(
                        model, temperature=temperature, **kwargs
                    )
                case 'deepseek':
                    self._current_llm = connect_deepseek_llm(
                        model, temperature=temperature, **kwargs
                    )
                case _:
                    raise ValueError(
                        f'<connect_llm> 不支持 {platform} 平台的 {model}！！！'
                    )

            self._current_llm_platform_name = platform
            self._current_llm_model_name = model
            self.llm_bind_tools()
        except Exception:
            logger.error(
                f'<connect_llm> 连接 {platform} 平台的 {model} 失败！！！\n{format_exc()}'
            )
            self.clean_llm()
            raise

    def set_llm_tools(self, tools: list):
        """设置 LLM 工具"""

        self._current_llm_tools = tools
        if self._current_llm is not None:
            self.llm_bind_tools()

    def get_llm(self) -> BaseChatModel:
        """获取 LLM"""

        if not self._current_llm_bind_tools:
            error_message = '<get_llm> 当前未连接 LLM！！！'
            logger.error(error_message)
            raise ValueError(error_message)

        return self._current_llm_bind_tools
