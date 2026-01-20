import os
import sys
import re
from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent
readme_path = here / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

def get_version():
    ver_path = here / "opt" / "__init__.py"
    if not ver_path.exists():
        return "0.1.0"
    text = ver_path.read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', text, re.M)
    return m.group(1) if m else "0.1.0"

def get_install_requires():
    requires = [
        "onnx>=1.15.0",
        "setuptools>=42.0.0",
        "numpy",
        "onnx-graphsurgeon>=0.4.0",
    ]
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
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    include_package_data=True,
    install_requires=get_install_requires(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "build>=0.10.0",
            "twine>=4.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
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
    zip_safe=False,
)
