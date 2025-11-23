"""
Ollama backend for local LLM inference.
Supports any model via Ollama (Qwen, DeepSeek, Llama, etc.)
"""

import json
import subprocess
import requests
from typing import Dict, List, Optional, Any
from cerebro.utils.logging import get_logger

logger = get_logger(__name__)


class OllamaBackend:
    """
    Backend for communicating with Ollama-served LLMs.
    
    Handles:
    - Chat completion
    - Function calling
    - Reasoning (CoT)
    - Code generation
    """
    
    def __init__(
        self,
        model: str = "deepseek-coder-v2:16b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 300
    ):
        """
        Initialize Ollama backend.
        
        Args:
            model: Model name (e.g., "qwen2.5-coder:32b")
            base_url: Ollama API URL
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        self._check_ollama_running()
        self._ensure_model_available()
    
    def _check_ollama_running(self):
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Ollama is not running or not accessible at {self.base_url}. "
                f"Please start Ollama: 'ollama serve'\nError: {e}"
            )
    
    def _ensure_model_available(self):
        """Check if model is available, pull if needed."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m['name'] for m in response.json()['models']]
            
            if self.model not in models:
                logger.info(f"Model {self.model} not found. Pulling...")
                self._pull_model()
        except Exception as e:
            raise RuntimeError(f"Failed to check model availability: {e}")
    
    def _pull_model(self):
        """Pull model from Ollama registry."""
        logger.info(f"Pulling {self.model}... This may take several minutes.")
        result = subprocess.run(
            ["ollama", "pull", self.model],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to pull model: {result.stderr}")
        
        logger.info(f"Successfully pulled {self.model}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Assistant's response text
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")
    
    def reason(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Chain-of-thought reasoning.
        
        Args:
            prompt: Question or problem to reason about
            context: Optional context information
            
        Returns:
            Reasoning output
        """
        system = (
            "You are an expert data scientist. Think step-by-step. "
            "Reason carefully about the problem before providing a solution."
        )
        
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        
        return self.chat(messages)
    
    def generate_code(
        self,
        task: str,
        data_info: Dict[str, Any],
        requirements: Optional[List[str]] = None
    ) -> str:
        """
        Generate Python code for a data science task.
        
        Args:
            task: Description of what to do
            data_info: Information about the data
            requirements: Specific requirements or constraints
            
        Returns:
            Generated Python code
        """
        system = (
            "You are an expert Python data scientist. Generate complete, "
            "executable Python code. Do not include explanations, only code. "
            "Use pandas, numpy, scipy, scikit-learn, statsmodels as needed."
        )
        
        # Convert numpy types and DataFrames to JSON-serializable types
        def convert_numpy(obj):
            """Recursively convert numpy/pandas types to Python native types."""
            import numpy as np
            import pandas as pd
            import types
            
            if isinstance(obj, types.ModuleType):
                # Skip modules
                return f"<module '{obj.__name__}'>"
            elif isinstance(obj, pd.DataFrame):
                # Convert DataFrame to description (don't serialize the whole thing!)
                return f"DataFrame({obj.shape[0]} rows × {obj.shape[1]} cols)"
            elif isinstance(obj, pd.Series):
                return f"Series(length={len(obj)})"
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy(item) for item in obj)
            elif isinstance(obj, (np.bool_, np.bool8)):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float_, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_numpy(obj.tolist())
            elif callable(obj):
                # Skip functions/callables
                return f"<callable '{getattr(obj, '__name__', 'unknown')}'>"
            return obj
        
        data_info_clean = convert_numpy(data_info)
        
        prompt = f"""Task: {task}

Data Information:
{json.dumps(data_info_clean, indent=2)}
"""
        
        if requirements:
            prompt += f"\nRequirements:\n" + "\n".join(f"- {r}" for r in requirements)
        
        prompt += "\n\nGenerate complete Python code:"
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        
        code = self.chat(messages, temperature=0.0)
        
        # Extract code from markdown if present
        code = self._extract_code_from_markdown(code)
        
        return code
    
    def function_call(
        self,
        prompt: str,
        available_functions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Function calling - LLM decides which function to call.
        
        Args:
            prompt: User request
            available_functions: List of function definitions
            
        Returns:
            Dict with 'function' and 'arguments'
        """
        system = (
            "You are a function calling assistant. Given a user request and "
            "available functions, decide which function to call and what arguments to use. "
            "Respond with JSON: {\"function\": \"function_name\", \"arguments\": {\"arg1\": \"value1\"}}"
        )
        
        functions_str = json.dumps(available_functions, indent=2)
        full_prompt = f"Available functions:\n{functions_str}\n\nUser request: {prompt}"
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": full_prompt}
        ]
        
        response = self.chat(messages, temperature=0.0)
        
        # Parse JSON response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            raise ValueError(f"Failed to parse function call response: {response}")
    
    def _extract_code_from_markdown(self, text: str) -> str:
        """Extract code from markdown code blocks."""
        import re
        
        # Try to extract from ```python blocks
        pattern = r'```python\n(.*?)\n```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Try to extract from ``` blocks
        pattern = r'```\n(.*?)\n```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Return as-is if no markdown
        return text
    
    def interpret_results(
        self,
        task: str,
        results: Dict[str, Any],
        business_context: Optional[str] = None
    ) -> str:
        """
        Interpret analysis results in business terms.
        
        Args:
            task: What analysis was performed
            results: Analysis results (numbers, statistics, etc.)
            business_context: Optional business context
            
        Returns:
            Plain English interpretation
        """
        system = (
            "You are a data science communicator. Explain technical results "
            "in clear business terms. Be specific, actionable, and honest about "
            "limitations."
        )
        
        prompt = f"""Task performed: {task}

Results:
{json.dumps(results, indent=2)}
"""
        
        if business_context:
            prompt += f"\nBusiness Context: {business_context}"
        
        prompt += "\n\nProvide a clear interpretation with actionable insights:"
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        
        return self.chat(messages)

