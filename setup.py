"""
Editable-install shim so `import lma` works from anywhere in the repo
(local machine, Kaggle notebook, CI) without sys.path hacks.

Usage:
    pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="lma",
    version="0.1.0",
    description="Shared library for two independent monolingual Transformer LMs (Hindi, Nepali) — CL3410 Individual Project",
    packages=find_packages(include=["lma", "lma.*"]),
    python_requires=">=3.10",
)
