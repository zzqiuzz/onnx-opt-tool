import os
import sys

from setuptools import setup, find_packages
from pathlib import Path

readme_path = Path(__file__).with_name("README.md")
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
opt_path = Path(__file__).with_name("opt")

def get_version(): 
    version_file = os.path.join(opt_path, "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line.startswith("__version__"):
                return line.strip().split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

def get_install_requires():
    """动态生成安装依赖列表"""
    requires = [
        "onnx>=1.15.0",          # 核心依赖 
        "setuptools>=42.0.0",    # 打包工具依赖
        "numpy",
        "onnx-graphsurgeon>=0.4.0"
    ]
    # 对Python 3.8及以下版本添加额外依赖
    if sys.version_info < (3, 9):
        requires.append("importlib-metadata>=4.0.0")
    return requires

setup(
    name="onnx_opt",
    version=get_version(),
    description="ONNX optimization utilities and tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="lepaulski",
    author_email="zzqiuzz@gmail.com",
    packages=find_packages(
        exclude=["tests*", "docs*", "examples*"]
    ),
    include_package_data=True,
    install_requires=get_install_requires(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",    # 测试框架
            "flake8>=6.0.0",    # 代码检查
            "black>=23.0.0",    # 代码格式化
            "build>=0.10.0",    # 打包工具
            "twine>=4.0.0",     # 上传PyPI工具
        ],
        "docs": [
            "sphinx>=7.0.0",    # 文档生成
            "sphinx-rtd-theme>=1.0.0",
        ]
    },
    entry_points={
        # 自定义插件命名空间（建议用项目相关的唯一名称）
        "mo_onnx_plugins": [
            "custom_onnx_optimizer = your_onnx_optimizer.core:ONNXGraphOptimizer",
        ],
        # 2. 控制台脚本（命令行直接调用）
        "console_scripts": [
            "onnx-optimize = your_onnx_optimizer.cli:main",
        ],
        # 3. 其他框架的扩展入口（如OpenVINO MO）
        "openvino_mo_extensions": [
            "onnx_optimizer_pass = your_onnx_optimizer.mo_adapter:register_extensions",
        ]
    },
    python_requires=">=3.8",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3", 
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
