import os
import logging
from logging.handlers import RotatingFileHandler


def setup_global_logging():
    """配置全局日志：同时输出到控制台和文件，统一格式"""
    # 1. 创建主logger（名称为工程名，所有模块的logger会继承它的配置）
    logger = logging.getLogger("onnx_opt_project")
    logger.setLevel(logging.DEBUG)  # 全局日志级别（最低级别，子模块可覆盖）

    # 避免重复添加handler（多次运行时）
    if logger.handlers:
        return logger

    # 2. 定义日志格式（包含时间、模块、级别、消息）
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s"
        # 新增字段：
        # %(funcName)s：调用日志的函数名
    )

    # 3. 控制台Handler（输出INFO及以上级别）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 4. 文件Handler（输出DEBUG及以上级别，支持日志滚动）
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # 创建日志目录
    file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, "app.log"),
        maxBytes=1024 * 1024,  # 单个日志文件最大1MB
        backupCount=3,         # 最多保留3个备份文件
        encoding="utf-8"       # 支持中文
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 5. 给主logger添加handler
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

__all__ = ["setup_global_logging"]