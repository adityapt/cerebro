"""
Auto Backend - Automatically detects environment and uses optimal inference engine

Mac M1/M2/M3    → MLX (5x faster than Ollama)
GPU Server      → vLLM (2-5x faster than Ollama)
CPU/Fallback    → Ollama (universal compatibility)
"""

import platform
import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def detect_environment():
    """
    Detect the current hardware environment.
    
    Returns:
        str: 'mac_silicon', 'nvidia_gpu', 'cpu', or 'unknown'
    """
    system = platform.system()
    machine = platform.machine()
    
    # Check for Apple Silicon
    if system == "Darwin" and machine in ["arm64", "aarch64"]:
        logger.info("Detected: Apple Silicon (M1/M2/M3)")
        return "mac_silicon"
    
    # Check for NVIDIA GPU
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.info("Detected: NVIDIA GPU")
            return "nvidia_gpu"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check for CUDA availability (alternative method)
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("Detected: NVIDIA GPU (via PyTorch)")
            return "nvidia_gpu"
    except ImportError:
        pass
    
    # Fallback to CPU
    logger.info(f"Detected: CPU only ({system} {machine})")
    return "cpu"


def get_optimal_backend(
    model: str = "qwen2.5-coder:32b",
    force_backend: Optional[str] = None,
    **kwargs
):
    """
    Get the optimal LLM backend for the current environment.
    
    Auto-detection logic:
    - Mac M1/M2/M3 → MLX (10-15 tok/s)
    - NVIDIA GPU   → vLLM (50-100 tok/s)
    - CPU/Other    → Ollama (2-5 tok/s)
    
    Args:
        model: Model name (format depends on backend)
        force_backend: Force specific backend ('mlx', 'vllm', 'ollama')
        **kwargs: Backend-specific parameters
        
    Returns:
        Initialized backend instance
    """
    # Allow manual override
    if force_backend:
        logger.info(f"Forcing backend: {force_backend}")
        env_type = force_backend
    else:
        env_type = detect_environment()
    
    # Select and initialize backend
    if env_type == "mac_silicon" or force_backend == "mlx":
        return _init_mlx_backend(model, **kwargs)
    
    elif env_type == "nvidia_gpu" or force_backend == "vllm":
        return _init_vllm_backend(model, **kwargs)
    
    else:  # cpu or unknown
        return _init_ollama_backend(model, **kwargs)


def _init_mlx_backend(model: str, **kwargs):
    """Initialize MLX backend for Apple Silicon"""
    try:
        from cerebro.llm.mlx_backend import MLXBackend
        
        # Store original model name for fallback
        original_model = model
        
        # Convert model name if needed
        if not model.startswith("mlx-community/"):
            # Map common model names to MLX versions
            model_map = {
                "qwen2.5-coder:32b": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
                "qwen2.5-coder:14b": "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
                "qwen2.5-coder:7b": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
            }
            model = model_map.get(model, model)
        
        logger.info(f"✅ Using MLX backend (optimized for Apple Silicon)")
        return MLXBackend(model=model, **kwargs)
        
    except ImportError as e:
        logger.warning(f"MLX not available: {e}")
        logger.info("Falling back to Ollama...")
        # Use original model name for fallback (not the MLX-converted one)
        return _init_ollama_backend(original_model, **kwargs)


def _init_vllm_backend(model: str, **kwargs):
    """Initialize vLLM backend for NVIDIA GPUs"""
    try:
        from cerebro.llm.vllm_backend import vLLMBackend
        
        # Convert model name if needed
        if ":" in model:  # Ollama-style name
            model_map = {
                "qwen2.5-coder:32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "qwen2.5-coder:14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
                "qwen2.5-coder:7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
            }
            model = model_map.get(model, model.split(":")[0])
        
        logger.info(f"✅ Using vLLM backend (optimized for NVIDIA GPU)")
        return vLLMBackend(model=model, **kwargs)
        
    except ImportError as e:
        logger.warning(f"vLLM not available: {e}")
        logger.info("Falling back to Ollama...")
        return _init_ollama_backend(model, **kwargs)


def _init_ollama_backend(model: str, **kwargs):
    """Initialize Ollama backend (universal fallback)"""
    from cerebro.llm.ollama_backend import OllamaBackend
    
    # Convert model name if needed
    if "/" in model and ":" not in model:  # HF format
        model_map = {
            "Qwen/Qwen2.5-Coder-32B-Instruct": "qwen2.5-coder:32b",
            "Qwen/Qwen2.5-Coder-14B-Instruct": "qwen2.5-coder:14b",
            "Qwen/Qwen2.5-Coder-7B-Instruct": "qwen2.5-coder:7b",
        }
        model = model_map.get(model, model)
    
    logger.info(f"✅ Using Ollama backend (universal compatibility)")
    return OllamaBackend(model=model, **kwargs)


class AutoBackend:
    """
    Automatic backend selection with environment detection.
    
    Usage:
        # Auto-detect (recommended)
        llm = AutoBackend(model="qwen2.5-coder:32b")
        
        # Force specific backend
        llm = AutoBackend(model="qwen2.5-coder:32b", force_backend="mlx")
        
    The AutoBackend will automatically use:
    - MLX on Mac M1/M2/M3 (10-15 tok/s)
    - vLLM on GPU servers (50-100 tok/s)
    - Ollama on CPU/other (2-5 tok/s)
    """
    
    def __init__(
        self,
        model: str = "qwen2.5-coder:32b",
        force_backend: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize auto-detecting backend.
        
        Args:
            model: Model name
            force_backend: Force specific backend ('mlx', 'vllm', 'ollama')
            **kwargs: Backend-specific parameters
        """
        self.backend = get_optimal_backend(
            model=model,
            force_backend=force_backend,
            **kwargs
        )
        
        # Expose backend methods
        self.reason = self.backend.reason
        self.generate = self.backend.generate
        self.chat = self.backend.chat
        self.generate_code = self.backend.generate_code
        self.get_stats = self.backend.get_stats
    
    def __repr__(self):
        return f"AutoBackend({self.backend.__class__.__name__})"

