import logging
from typing import Optional

class Logger:
    _instance: Optional['Logger'] = None
    _logger: logging.Logger

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialize_logger()
        return cls._instance

    @classmethod
    def _initialize_logger(cls):
        cls._logger = logging.getLogger("ONNXOptimizer")
        cls._logger.setLevel(logging.INFO)

        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        cls._logger.addHandler(ch)

    def set_level(self, level: int):
        self._logger.setLevel(level)

    def debug(self, message: str):
        self._logger.debug(message)

    def info(self, message: str):
        self._logger.info(message)

    def warning(self, message: str):
        self._logger.warning(message)

    def error(self, message: str):
        self._logger.error(message)

# 单例导出
logger = Logger()