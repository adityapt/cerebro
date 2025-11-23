"""LLM backends for Cerebro - Adaptive multi-platform support"""

from cerebro.llm.ollama_backend import OllamaBackend
from cerebro.llm.hybrid_backend import HybridBackend
from cerebro.llm.rag_backend import RAGBackend
from cerebro.llm.qwen_rag_backend import QwenRAGBackend
from cerebro.llm.tree_of_thought import TreeOfThought, GraphOfThought
from cerebro.llm.code_judge import CodeJudge
from cerebro.llm.results_judge import ResultsJudge

# Platform-optimized backends
from cerebro.llm.mlx_backend import MLXBackend
from cerebro.llm.vllm_backend import vLLMBackend
from cerebro.llm.auto_backend import AutoBackend, get_optimal_backend

__all__ = [
    "OllamaBackend",
    "HybridBackend",
    "RAGBackend",
    "QwenRAGBackend",
    "TreeOfThought",
    "GraphOfThought",
    "CodeJudge",
    "ResultsJudge",
    # Platform-optimized
    "MLXBackend",
    "vLLMBackend",
    "AutoBackend",
    "get_optimal_backend"
]
