"""
MLX Backend - Optimized for Apple Silicon (M1/M2/M3)
5x faster than Ollama on Mac
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class MLXBackend:
    """
    LLM backend using Apple's MLX framework.
    
    Optimized for:
    - Apple Silicon (M1/M2/M3/M4)
    - Metal GPU acceleration
    - Unified memory architecture
    
    Speed: 10-15 tok/s on M2 (5x faster than Ollama)
    """
    
    def __init__(
        self,
        model: str = "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
        max_tokens: int = 4096,
        temperature: float = 0.1
    ):
        """
        Initialize MLX backend.
        
        Args:
            model: Model name from mlx-community on Hugging Face
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        try:
            from mlx_lm import load, generate, stream_generate
            self.mlx_load = load
            self.mlx_generate = generate
            self.mlx_stream_generate = stream_generate
        except ImportError:
            raise ImportError(
                "MLX not installed. Install with: pip install mlx-lm\n"
                "Note: MLX only works on Apple Silicon (M1/M2/M3)"
            )
        
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initializing MLX backend with {model}")
        self._load_model()
    
    def _load_model(self):
        """Load model using MLX (optimized for Apple Silicon)"""
        logger.info("Loading model with MLX (Metal acceleration)...")
        try:
            self.model, self.tokenizer = self.mlx_load(self.model_name)
            logger.info("✅ MLX model loaded successfully!")
            logger.info(f"   Model: {self.model_name}")
            logger.info(f"   Device: Apple Metal GPU")
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}")
            raise
    
    def reason(self, prompt: str, **kwargs) -> str:
        """
        Generate response using MLX with streaming output.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
                     stream: bool - whether to stream output (default True)
            
        Returns:
            Generated text
        """
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        stream = kwargs.get('stream', True)
        
        # Apply chat template for Qwen models
        # This formats the prompt correctly for instruction-following
        try:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": "You are a helpful coding assistant. Generate clean, working Python code."},
                    {"role": "user", "content": prompt}
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback for models without chat template
                formatted_prompt = prompt
        except Exception as e:
            logger.warning(f"Chat template application failed, using raw prompt: {e}")
            formatted_prompt = prompt
        
        try:
            import sys
            
            if stream:
                # Stream tokens as they're generated
                print("\n" + "="*80)
                print("🤖 GENERATING CODE (watch it appear line-by-line):")
                print("="*80 + "\n")
                
                full_response = ""
                for response in self.mlx_stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=max_tokens
                ):
                    # response.text contains the next segment (incremental)
                    # Accumulate and print immediately
                    if response.text:
                        print(response.text, end='', flush=True)
                        full_response += response.text
                
                print("\n\n" + "="*80)
                print("✅ CODE GENERATION COMPLETE")
                print("="*80 + "\n")
                return full_response
            else:
                # Non-streaming (silent)
                response = self.mlx_generate(
                    self.model,
                    self.tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    verbose=False
                )
                return response
        except Exception as e:
            logger.error(f"MLX generation failed: {e}")
            raise RuntimeError(f"MLX generation error: {e}")
    
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
            'backend': 'MLX',
            'model': self.model_name,
            'device': 'Apple Metal GPU',
            'optimized_for': 'Apple Silicon (M1/M2/M3)',
            'expected_speed': '10-15 tok/s'
        }

