from setuptools import setup, find_packages

setup(
    name="multi_agent_system",
    version="0.1.0",
    packages=find_packages(include=["multi_agent_system", "multi_agent_system.*"]),
    install_requires=[
        "langchain",
        "openai",
        "pandas",
        "protollm",
        "requests",
        "gradio",
        "langgraph"
    ],
    description="MADD package",
    classifiers=[
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
)
