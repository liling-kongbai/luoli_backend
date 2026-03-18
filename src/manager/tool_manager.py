from logging import getLogger
from traceback import format_exc

from langchain_core.tools.base import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = getLogger(__name__)

SAFE_TOOLS_NAME_LIST = ['search', 'weather']


class ToolManager:
    """工具管理器"""

    def __init__(self, safe_tools_name_list: list[str] | None = SAFE_TOOLS_NAME_LIST):
        self._safe_tools_name_list: list[str] | None = safe_tools_name_list
        self._tools: list[BaseTool] | None = None
        self._safe_tools: list[BaseTool] | None = None
        self._mcp_clinet: MultiServerMCPClient | None = None

    def clean(self):
        """清理"""

        self._safe_tools = None
        self._tools = None
        self._mcp_clinet = None

    def create_mcp_client(self, args_path: str, cwd_path: str):
        """创建 MCP 客户端"""

        if self._mcp_clinet:
            logger.warning(
                '<create_mcp_client> MCP 客户端已创建，请勿重复创建，请检查代码逻辑！！！'
            )
            return

        try:
            self._mcp_clinet = MultiServerMCPClient(
                {
                    'test': {
                        'transport': 'stdio',
                        'command': 'uv',
                        'args': ['run', args_path],
                        'cwd': cwd_path,
                    }
                }
            )
        except Exception:
            logger.error(
                f'<create_mcp_client> 创建 MCP 客户端报错！！！\n{format_exc()}'
            )
            raise

    async def get_mcp_tools(self):
        """获取 MCP 工具"""

        if not self._mcp_clinet:
            logger.warning('<get_mcp_tools> MCP 客户端未创建，请检查代码逻辑！！！')
            return

        try:
            self._tools.extend(await self._mcp_clinet.get_tools())
        except Exception:
            logger.error(f'<get_mcp_tools> 获取 MCP 工具报错！！！\n{format_exc()}')
            raise

    def register_safe_tools(self):
        """注册安全工具"""

        for tool in self._tools:
            if tool.name in self._safe_tools_name_list:
                self._safe_tools.append(tool)

    def get_tools(self) -> list[BaseTool]:
        """获取工具"""

        return self._tools

    def get_safe_tools(self) -> list[BaseTool]:
        """获取安全工具"""

        return self._safe_tools

    def get_tool(self, tool_name: str) -> BaseTool | None:
        """获取工具"""

        for tool in self._tools:
            if tool.name == tool_name:
                return tool
        return None
