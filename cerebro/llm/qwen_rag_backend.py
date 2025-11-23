"""
Qwen 32B + RAG Backend for Cerebro.

Combines:
1. RAG retrieval for relevant examples
2. Qwen 32B for code generation
3. Agentic flow with reflection
"""

import logging
from typing import List, Dict, Any, Optional
from cerebro.llm.ollama_backend import OllamaBackend
from cerebro.llm.rag_backend import RAGBackend

logger = logging.getLogger(__name__)


class QwenRAGBackend:
    """
    Production-grade backend combining Qwen 32B with RAG.
    
    Architecture:
    1. User query → RAG retrieves relevant examples
    2. Examples + query → Qwen 32B generates code
    3. Code → Validation & execution
    4. If errors → Reflection + regeneration
    """
    
    def __init__(
        self,
        model: str = "qwen2.5-coder:32b",
        dataset_path: str = "fine_tuning/rag_complete_real_synthetic.jsonl",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 300,
        n_rag_examples: int = 3
    ):
        """
        Initialize Qwen 32B + RAG backend.
        
        Args:
            model: Qwen model name
            dataset_path: Path to RAG dataset
            base_url: Ollama API URL
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            timeout: Request timeout
            n_rag_examples: Number of examples to retrieve
        """
        logger.info("Initializing Qwen 32B + RAG backend...")
        
        # Initialize Qwen 32B backend
        logger.info(f"Initializing Qwen 32B: {model}")
        self.llm = OllamaBackend(
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
        
        # Initialize RAG
        logger.info("Initializing RAG system...")
        self.rag = RAGBackend(dataset_path=dataset_path)
        
        self.n_rag_examples = n_rag_examples
        
        logger.info("✅ Qwen 32B + RAG backend ready!")
        logger.info(f"   Model: {model}")
        logger.info(f"   RAG examples: {self.rag.get_stats()['total_examples']}")
    
    def generate(self, prompt: str, use_rag: bool = True, **kwargs) -> str:
        """
        Generate text completion with optional RAG augmentation.
        
        Args:
            prompt: Input prompt
            use_rag: Whether to use RAG for retrieval
            **kwargs: Additional arguments for LLM
        
        Returns:
            Generated text
        """
        if use_rag:
            # Extract query from prompt for RAG
            query = self._extract_query(prompt)
            
            # Augment prompt with RAG
            augmented_prompt = self.rag.augment_prompt(
                base_prompt=prompt,
                query=query,
                n_examples=self.n_rag_examples
            )
            
            logger.debug(f"Augmented prompt with {self.n_rag_examples} RAG examples")
            return self.llm.reason(prompt=augmented_prompt)
        else:
            return self.llm.reason(prompt=prompt)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        use_rag: bool = True,
        **kwargs
    ) -> str:
        """
        Generate chat completion with optional RAG augmentation.
        
        Args:
            messages: List of chat messages
            use_rag: Whether to use RAG for retrieval
            **kwargs: Additional arguments for LLM
        
        Returns:
            Generated response
        """
        if use_rag and messages:
            # Extract query from last message
            last_message = messages[-1]['content']
            query = self._extract_query(last_message)
            
            # Retrieve examples
            examples = self.rag.retrieve(query, n_results=self.n_rag_examples)
            
            if examples:
                # Add examples as a system message
                examples_text = self.rag.format_retrieved_examples(
                    examples,
                    max_length=2000
                )
                
                # Insert examples before user message
                augmented_messages = messages[:-1] + [
                    {
                        'role': 'system',
                        'content': f"Here are relevant examples:\n\n{examples_text}"
                    },
                    messages[-1]
                ]
                
                # Flatten messages to single prompt for reason()
                flattened = "\n".join([f"{m['role']}: {m['content']}" for m in augmented_messages])
                return self.llm.reason(prompt=flattened)
        
        # Flatten messages to single prompt
        flattened = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return self.llm.reason(prompt=flattened)
    
    def generate_code(
        self,
        task: str,
        data_info: Dict[str, Any],
        requirements: Optional[List[str]] = None,
        use_rag: bool = True,
        **kwargs
    ) -> str:
        """
        Generate code with RAG augmentation.
        
        Args:
            task: Task description
            data_info: Information about the data
            requirements: Additional requirements
            use_rag: Whether to use RAG
            **kwargs: Additional arguments for LLM
        
        Returns:
            Generated code
        """
        # Create base prompt
        prompt = self._create_code_prompt(task, data_info, requirements)
        
        # Generate with RAG
        response = self.generate(prompt, use_rag=use_rag, **kwargs)
        
        # Extract code from response
        code = self._extract_code(response)
        
        return code
    
    def _extract_query(self, text: str) -> str:
        """
        Extract a search query from a prompt or message.
        
        For RAG retrieval, we want to extract the core task/concept.
        """
        # Simple extraction: use the text as-is
        # Could be enhanced with keyword extraction, etc.
        return text[:500]  # Limit length
    
    def _create_code_prompt(
        self,
        task: str,
        data_info: Dict[str, Any],
        requirements: Optional[List[str]] = None
    ) -> str:
        """Create a prompt for code generation"""
        prompt = f"""You are an expert Python programmer specializing in Marketing Data Science.

Task: {task}

Data Information:
"""
        # Add data info
        for key, value in data_info.items():
            if isinstance(value, (list, dict)):
                prompt += f"- {key}: {value}\n"
            else:
                prompt += f"- {key}: {value}\n"
        
        if requirements:
            prompt += "\nRequirements:\n"
            for req in requirements:
                prompt += f"- {req}\n"
        
        prompt += "\nGenerate complete, production-grade Python code (code only, no explanations):"
        
        return prompt
    
    def _extract_code(self, response: str) -> str:
        """Extract code from LLM response"""
        if "```python" in response:
            return response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            return response.split("```")[1].split("```")[0].strip()
        return response.strip()
    
    def reason(self, prompt: str, **kwargs) -> str:
        """Alias for generate() - used by agents"""
        return self.generate(prompt, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        return {
            'model': self.llm.model,
            'rag_stats': self.rag.get_stats(),
            'n_rag_examples': self.n_rag_examples
        }

