"""
Basic import tests for Cerebro package.
"""

import pytest


def test_cerebro_import():
    """Test that main Cerebro class can be imported."""
    from cerebro import Cerebro
    assert Cerebro is not None


def test_ollama_backend_import():
    """Test that OllamaBackend can be imported."""
    from cerebro.llm.ollama_backend import OllamaBackend
    assert OllamaBackend is not None


def test_tree_of_thought_import():
    """Test that TreeOfThought can be imported."""
    from cerebro.llm.tree_of_thought import TreeOfThought, GraphOfThought
    assert TreeOfThought is not None
    assert GraphOfThought is not None


def test_code_judge_import():
    """Test that CodeJudge can be imported."""
    from cerebro.llm.code_judge import CodeJudge
    assert CodeJudge is not None


def test_logging_import():
    """Test that logging utilities can be imported."""
    from cerebro.utils.logging import get_logger, setup_logger
    assert get_logger is not None
    assert setup_logger is not None

