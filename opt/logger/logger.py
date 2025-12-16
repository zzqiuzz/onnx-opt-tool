import os
import logging
from logging.handlers import RotatingFileHandler


def setup_global_logging():
    """配置全局日志：同时输出到控制台和文件，格式对齐整洁

    将处理器安装到根日志器（root logger），确保使用
    `logging.getLogger(__name__)` 的模块能够继承并输出日志。

    格式优化点：
    1. 时间字段固定格式和长度（含毫秒）
    2. 日志级别固定8字符宽度（左对齐）
    3. 日志器名称固定15字符宽度（左对齐）
    4. 文件名固定20字符宽度（左对齐）
    5. 函数名固定20字符宽度（左对齐）
    6. 行号固定4位数字（右对齐，不足补空格）
    """
    # 1. 使用根logger，所有模块的logger都会向根logger传播
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 全局最低日志级别

    # 避免重复添加handler（多次调用该函数时）
    if logger.handlers:
        return logger

    # 2. 定义统一的时间格式（固定长度：23字符，含毫秒）
    date_format = "%Y-%m-%d %H:%M:%S,%f"[:-3]  # 保留毫秒（去掉微秒后三位）

    # 3. 定义对齐的日志格式
    # 各字段宽度说明：
    # - levelname:8  → 级别占8字符（DEBUG/INFO/WARNING等都对齐）
    # - name:15      → 日志器名称占15字符（模块名不会错位）
    # - filename:20  → 文件名占20字符
    # - funcName:20  → 函数名占20字符
    # - lineno:4d    → 行号占4位数字（右对齐，如  123、1234）
    log_format = (
        "%(asctime)s - %(levelname)-5s - %(name)-20s "
        "- [%(filename)-20s:%(funcName)-10s:%(lineno)4d] - %(message)s"
    )

    # 控制台和文件使用相同的对齐格式（可根据需要单独调整）
    console_formatter = logging.Formatter(log_format, date_format)
    file_formatter = logging.Formatter(log_format, date_format)

    # 4. 控制台Handler（输出INFO及以上级别）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # 5. 文件Handler（输出DEBUG及以上级别，支持日志滚动）
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # 确保日志目录存在
    file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, "app.log"),
        maxBytes=1024 * 1024,  # 单个日志文件最大1MB
        backupCount=3,         # 最多保留3个备份文件
        encoding="utf-8"       # 支持中文日志
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # 6. 添加处理器到根logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


__all__ = ["setup_global_logging"]