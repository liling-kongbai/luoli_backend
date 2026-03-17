from contextlib import asynccontextmanager
from logging import getLogger
from traceback import format_exc

from fastapi import FastAPI

from .set_async_logger import set_async_logger
from .src.agent import Agent

# 日志相关
log_file_path = r'logs/luo_li_backend.log'
log_listener = set_async_logger(log_file_path=log_file_path)
logger = getLogger(__name__)
logger.info(f'<main.py> 异步日志器已设置并启动，日志文件路径：{log_file_path}')


agent: Agent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """异步上下文管理器，应用生命周期管理器，让 FastAPI 自动运行应用核心服务的启动和关闭"""

    try:
        global agent
        agent = Agent()

        yield

        agent = None
    except Exception:
        logger.error(f'<lifespan> 应用生命周期管理器报错！！！\n{format_exc()}')
    finally:
        log_listener.stop()


app = FastAPI(lifespan=lifespan)
