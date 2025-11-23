"""
vLLM Backend - Optimized for NVIDIA GPUs (Databricks, AWS, GCP, Azure)
2-5x faster than Ollama on GPU servers
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class vLLMBackend:
    """
    LLM backend using vLLM for high-performance GPU inference.
    
    Optimized for:
    - NVIDIA GPUs (A100, A10G, T4, etc.)
    - Cloud environments (Databricks, AWS, GCP, Azure)
    - High-throughput inference
    
    Speed: 50-100 tok/s on A100 (2-5x faster than Ollama)
    """
    
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        download_dir: Optional[str] = None
    ):
        """
        Initialize vLLM backend.
        
        Args:
            model: Model name from Hugging Face
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            download_dir: Directory to cache models
        """
        try:
            from vllm import LLM, SamplingParams
            self.LLM = LLM
            self.SamplingParams = SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm\n"
                "Note: vLLM requires NVIDIA GPU with CUDA"
            )
        
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm = None
        
        logger.info(f"Initializing vLLM backend with {model}")
        self._load_model(
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            download_dir=download_dir
        )
    
    def _load_model(
        self,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        download_dir: Optional[str]
    ):
        """Load model using vLLM (optimized for NVIDIA GPUs)"""
        logger.info("Loading model with vLLM (CUDA acceleration)...")
        try:
            init_kwargs = {
                'model': self.model_name,
                'tensor_parallel_size': tensor_parallel_size,
                'gpu_memory_utilization': gpu_memory_utilization,
                'trust_remote_code': True
            }
            
            if download_dir:
                init_kwargs['download_dir'] = download_dir
            
            self.llm = self.LLM(**init_kwargs)
            
            logger.info("✅ vLLM model loaded successfully!")
            logger.info(f"   Model: {self.model_name}")
            logger.info(f"   Device: NVIDIA GPU (CUDA)")
            logger.info(f"   Tensor parallel: {tensor_parallel_size}")
        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            raise
    
    def reason(self, prompt: str, **kwargs) -> str:
        """
        Generate response using vLLM.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        temperature = kwargs.get('temperature', self.temperature)
        
        sampling_params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95
        )
        
        try:
            outputs = self.llm.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text
            return response
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            raise RuntimeError(f"vLLM generation error: {e}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Alias for reason()"""
        return self.reason(prompt, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Chat completion (converts messages to single prompt).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        # Convert messages to single prompt
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return self.reason(prompt, **kwargs)
    
    def generate_code(
        self,
        task: str,
        data_info: Dict[str, Any],
        requirements: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate code for a specific task.
        
        Args:
            task: Task description
            data_info: Information about the data
            requirements: Optional list of requirements
            **kwargs: Additional parameters
            
        Returns:
            Generated code
        """
        prompt = self._create_code_prompt(task, data_info, requirements)
        response = self.reason(prompt, **kwargs)
        return self._extract_code(response)
    
    def _create_code_prompt(
        self,
        task: str,
        data_info: Dict[str, Any],
        requirements: Optional[List[str]] = None
    ) -> str:
        """Create prompt for code generation"""
        prompt = f"Task: {task}\n\n"
        prompt += f"Data info: {data_info}\n\n"
        
        if requirements:
            prompt += "Requirements:\n"
            for req in requirements:
                prompt += f"- {req}\n"
            prompt += "\n"
        
        prompt += "Generate Python code:\n```python\n"
        return prompt
    
    def _extract_code(self, response: str) -> str:
        """Extract code from response"""
        # Try to extract from code blocks
        if "```python" in response:
            parts = response.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()
        
        if "```" in response:
            parts = response.split("```")
            if len(parts) > 1:
                return parts[1].strip()
        
        # Return as-is if no code blocks
        return response.strip()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        return {
            'backend': 'vLLM',
            'model': self.model_name,
            'device': 'NVIDIA GPU (CUDA)',
            'optimized_for': 'Cloud GPU (Databricks, AWS, GCP, Azure)',
            'expected_speed': '50-100 tok/s'
        }


