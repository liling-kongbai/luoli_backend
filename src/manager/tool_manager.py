from logging import getLogger
from traceback import format_exc

from langchain_core.tools.base import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = getLogger(__name__)

SAFE_TOOLS_WHITELIST = [
    'search',
    'weather',
]


class ToolManager:
    """工具管理器"""

    def __init__(self, safe_tools_whitelist: list[str]):
        self._safe_tools_whitelist = safe_tools_whitelist
        self._tools: list[BaseTool] = []
        self._safe_tools: dict[str, BaseTool] = {}
        self._tool_is_safe: dict[str, bool] = {}
        self._mcp_server = None

    def register_and_tag_tools(self, tools: list[BaseTool]):
        """注册并标记工具"""

        for tool in tools:
            self._safe_tools[tool.name] = tool
            self._tool_is_safe[tool.name] = tool.name in self._safe_tools_whitelist

    def get_tools(self) -> list[BaseTool]:
        """获取工具"""

        return list[BaseTool](self._tools)

    def get_safe_tools(self) -> list[BaseTool]:
        """获取安全工具"""

        return list[BaseTool](self._safe_tools)

    def get_tool(self, tool_name: str) -> BaseTool:
        """获取工具"""

        return self._tools.get(tool_name)

    def create_mcp_client(self, path: str):
        """创建 MCP 客户端"""

        try:
            if self._mcp_server is not None:
                logger.warning('<connect_mcp_server> MCP 服务器已连接！！！')
                return

            self._mcp_server = MultiServerMCPClient(
                {
                    'test': {
                        'transport': 'stdio',
                        'command': 'uv',
                        'args': ['run', path],
                        'cwd': path,
                    }
                }
            )
        except Exception:
            logger.error(
                f'<connect_mcp_server> 连接 MCP 服务器失败！！！\n{format_exc()}'
            )
            raise

    async def get_mcp_tools(self) -> list[BaseTool]:
        """获取 MCP 工具"""

        try:
            if self._mcp_server is None:
                logger.warning('<get_mcp_tools> MCP 客户端未创建！！！')
                return []

            return self._tools.extend(await self._mcp_server.get_tools())
        except Exception:
            logger.error(f'<get_mcp_tools> 获取 MCP 工具失败！！！\n{format_exc()}')
            raise
