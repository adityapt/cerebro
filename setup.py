from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="cerebro-mmm",
    version="0.1.0",
    description="Autonomous Marketing Mix Modeling with Multi-Agent AI System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Cerebro Contributors",
    author_email="",
    url="https://github.com/yourusername/cerebro",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/cerebro/issues",
        "Source": "https://github.com/yourusername/cerebro",
        "Documentation": "https://github.com/yourusername/cerebro#readme",
    },
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "fine_tuning"]),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "PyYAML>=6.0",
        "jax[cpu]>=0.4.0",
        "numpyro>=0.13.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "statsmodels>=0.14.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "mlx": [
            "mlx-lm>=0.10.0",  # For Apple Silicon
        ],
        "full": [
            "mlx-lm>=0.10.0",
            "plotly>=5.14.0",
            "arviz>=0.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cerebro=cerebro.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Code Generators",
    ],
    keywords="marketing-mix-modeling mmm bayesian ai-agents code-generation rag llm",
)
