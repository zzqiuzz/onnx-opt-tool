from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Config:
    allow_overlap: bool = False  # 是否允许重叠匹配
    log_level: int = field(default=20)  # logging.INFO
    visualize: bool = False       # 是否可视化匹配结果

    def update(self, **kwargs: Any):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

# 默认配置
default_config = Config()