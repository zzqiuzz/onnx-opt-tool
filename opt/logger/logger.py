import os
import logging
from logging.handlers import RotatingFileHandler


def setup_global_logging():
    """配置全局日志：同时输出到控制台和文件，统一格式

    将处理器安装到根日志器（root logger），确保使用
    `logging.getLogger(__name__)` 的模块能够继承并输出日志。
    """
    # 1. 使用根logger，这样所有模块的logger都会向根logger传播
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 全局日志级别（最低级别，子模块可覆盖）

    # 避免重复添加handler（多次运行时）
    if logger.handlers:
        return logger

    # 2. 定义日志格式（包含时间、模块、级别、消息）
    # 使用固定宽度对齐 levelname，便于视觉上对齐 INFO/DEBUG/ERROR 等级别
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)7s - %(levelname)-7s - [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s"
    )

    # 文件中可以保留更宽一点的 level 字段（可选）
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)8s - %(levelname)-8s - [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s"
    )

    # 3. 控制台Handler（输出INFO及以上级别）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

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
    file_handler.setFormatter(file_formatter)

    # 5. 给主logger添加handler
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

__all__ = ["setup_global_logging"]