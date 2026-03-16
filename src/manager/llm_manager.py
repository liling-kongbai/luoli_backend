from logging import getLogger
from traceback import format_exc

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools.base import BaseTool

from ..utils import connect_deepseek_llm, connect_ollama_llm

logger = getLogger(__name__)


class LLMManager:
    """LLM 管理器"""

    def __init__(self):
        self._llm_platform_name: str | None = None
        self._llm_model_name: str | None = None
        self._tools: list[BaseTool] | None = None

        self._llm: BaseChatModel | None = None
        self._llm_bind_tools: BaseChatModel | None = None

        self._embedding_model = None

    def clean_llm(self):
        """清理 llm"""

        self._llm_bind_tools = None
        self._llm = None

    def llm_bind_tools(self):
        """LLM 绑定工具"""

        try:
            if self._tools:
                self._llm_bind_tools = self._llm.bind_tools(self._tools)
            else:
                self._llm_bind_tools = self._llm
        except Exception:
            self._llm_bind_tools = self._llm
            logger.error(f'<llm_bind_tools> LLM 绑定工具报错！！！\n{format_exc()}')
            raise

    def set_tools(self, tools: list[BaseTool]):
        """设置工具"""

        self._tools = tools
        if self._llm:
            self.llm_bind_tools()

    def connect_llm(
        self, platform: str, model: str, temperature: float | None = None, **kwargs
    ):
        """连接 LLM"""

        if not platform or not model:
            self.clean_llm()
            logger.info('<connect_llm> 平台或模型为空，LLM 已清理')
            return

        logger.info(f'<connect_llm> 连接 {platform} 平台的 {model}')
        try:
            match platform:
                case 'ollama':
                    self._llm = connect_ollama_llm(
                        model, temperature=temperature, **kwargs
                    )
                case 'deepseek':
                    self._llm = connect_deepseek_llm(
                        model, temperature=temperature, **kwargs
                    )
                case _:
                    raise ValueError(
                        f'<connect_llm> 不支持 {platform} 平台，请检查代码逻辑！！！'
                    )

            self._llm_platform_name = platform
            self._llm_model_name = model
            self.llm_bind_tools()
            logger.info(f'<connect_llm> 连接 {platform} 平台的 {model} 成功')
        except Exception:
            logger.error(f'<connect_llm> 连接 LLM 报错！！！\n{format_exc()}')
            self.clean_llm()
            raise

    def get_llm(self) -> BaseChatModel:
        """获取 LLM"""

        if not self._llm_bind_tools:
            logger.warning('<get_llm> 当前未连接 LLM，请检查代码逻辑！！！')
            return

        return self._llm_bind_tools
