"""
LLM API Backend - Supports OpenAI and OpenAI-compatible APIs
"""

import os
import yaml
import requests
from typing import Optional, Generator, Dict, Any


class ApiBackend:
    """
    LLM backend supporting OpenAI and OpenAI-compatible APIs.
    
    Supports multiple credential sources (in priority order):
    1. Direct parameters (api_key, base_url passed to __init__)
    2. Environment variables (OPENAI_API_KEY, OPENAI_BASE_URL, etc.)
    3. Config file (.api_config.yaml)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        config_path: str = ".api_config.yaml"
    ):
        """
        Initialize API backend with flexible credential sources.
        
        Args:
            api_key: API key (if None, reads from env or config)
            base_url: API base URL (if None, reads from env or config)
            model: Model name (if None, reads from env or config, defaults to 'gpt-4o')
            max_tokens: Max tokens (if None, reads from env or config, defaults to 4096)
            config_path: Path to YAML config file (used if params not provided)
        
        Priority:
            1. Direct parameters (highest priority)
            2. Environment variables (OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL)
            3. Config file (lowest priority)
        """
        # Try to load from parameters, env, or config file (in that order)
        self.api_key = (
            api_key or
            os.environ.get('OPENAI_API_KEY') or
            os.environ.get('API_KEY') or
            self._load_from_config(config_path, 'api_key')
        )
        
        self.base_url = (
            base_url or
            os.environ.get('OPENAI_BASE_URL') or
            os.environ.get('API_BASE_URL') or
            self._load_from_config(config_path, 'base_url') or
            'https://api.openai.com/v1'  # Default to OpenAI
        )
        
        self.model = (
            model or
            os.environ.get('OPENAI_MODEL') or
            os.environ.get('MODEL') or
            self._load_from_config(config_path, 'model') or
            'gpt-4o'
        )
        
        self.max_tokens = (
            max_tokens or
            int(os.environ.get('MAX_TOKENS', 0)) or
            self._load_from_config(config_path, 'max_tokens') or
            4096
        )
        
        if not self.api_key:
            raise ValueError(
                "API key not found. Please provide via:\n"
                "  1. api_key parameter: ApiBackend(api_key='sk-...')\n"
                "  2. Environment variable: export OPENAI_API_KEY='sk-...'\n"
                "  3. Config file: .api_config.yaml"
            )
        
        # Construct full endpoint
        self.endpoint = f"{self.base_url}/chat/completions"
    
    def _load_from_config(self, config_path: str, key: str):
        """Load a specific key from config file (if exists)."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Handle nested config structure
            if 'api' in config:
                return config['api'].get(key)
            else:
                return config.get(key)
        except (FileNotFoundError, KeyError, TypeError):
            return None
    
    def reason(self, prompt: str, stream: bool = False, max_tokens: Optional[int] = None):
        """
        Generate response from GPT-4o.
        
        Args:
            prompt: Input prompt
            stream: If True, return generator for streaming. If False, return string
            max_tokens: Override default max_tokens
            
        Returns:
            Generated text (str if stream=False, Generator if stream=True)
        """
        if stream:
            return self.stream_reason(prompt, max_tokens)
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': max_tokens or self.max_tokens,
            'temperature': 0.0,  # Deterministic for code generation
            'stream': False
        }
        
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=120
        )
        
        if response.status_code != 200:
            raise Exception(f"API error {response.status_code}: {response.text}")
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def stream_reason(self, prompt: str, max_tokens: Optional[int] = None) -> Generator[str, None, None]:
        """
        Stream response from GPT-4o token by token.
        
        Args:
            prompt: Input prompt
            max_tokens: Override default max_tokens
            
        Yields:
            Token strings
        """
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': max_tokens or self.max_tokens,
            'temperature': 0.0,  # Deterministic for code generation
            'stream': True
        }
        
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            stream=True,
            timeout=120
        )
        
        if response.status_code != 200:
            raise Exception(f"API error {response.status_code}: {response.text}")
        
        # Parse SSE stream
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        import json
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                yield delta['content']
                    except:
                        continue


# Backwards compatibility
class APIBackend(ApiBackend):
    """Alias for ApiBackend (case compatibility)"""
    pass

