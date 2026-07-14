# setup.py

from setuptools import setup, find_packages

setup(
    name="adt_triton_ops",
    version="0.1.0",  # 版本号
    description="compile ops to adt_triton_ops",
    author="wangsong95",
    author_email="wangsong95@126.com",
    url="https://github.com/Wangsong95/ascend-decoding-turbo",
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)