"""
Setup configuration for governance_pipeline package.
"""
from setuptools import setup, find_packages

setup(
    name="governance_pipeline",
    version="1.0.0",
    description="Governance Q&A Pipeline with LangGraph and Cortex",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "langgraph>=0.2.0",
        "langchain-core>=0.3.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "openpyxl>=3.1.0",
        "requests>=2.31.0",
    ],
)
