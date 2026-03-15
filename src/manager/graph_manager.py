from logging import getLogger
from traceback import format_exc

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from ..graph import create_inference_graph, create_main_graph, create_routine_graph

logger = getLogger(__name__)


class GraphManager:
    """图管理器"""

    def __init__(self, checkpoint_saver: BaseCheckpointSaver):
        self._checkpoint_saver: BaseCheckpointSaver | None = checkpoint_saver
        self._graph: CompiledStateGraph | None = None

    def clean(self):
        """清理"""

        self._graph = None
        self._checkpoint_saver = None
        logger.info('<clean> 图管理器已清理')

    def compile_graph(self):
        """编译图"""

        if self._graph:
            logger.warning(
                '<compile_graph> 图已存在，请勿重复编译，请检查代码逻辑！！！'
            )
            return

        if not self._checkpoint_saver:
            logger.warning(
                '<compile_graph> 未编译图，检查点保存器不存在，请检查代码逻辑！！！'
            )
            return

        try:
            logger.info('<compile_graph> 开始编译图')
            self._graph = create_main_graph(
                routine_graph=create_routine_graph(),
                inference_graph=create_inference_graph(),
                checkpoint_saver=self._checkpoint_saver,
            )
            logger.info('<compile_graph> 编译图完成')
        except Exception:
            logger.error(f'<compile_graph> 编译图失败！！！\n{format_exc()}')
            raise
