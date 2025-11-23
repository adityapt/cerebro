"""
Hybrid Backend: DeepSeek Model + DeepAnalyze Approach

This combines the best of both:
- DeepSeek 16B: Fast, reliable, works on M1
- DeepAnalyze style: Action tokens, reflection, agentic prompting

Action tokens (inspired by DeepAnalyze):
- ⟨Analyze⟩: Planning, reasoning, reflection
- ⟨Code⟩: Generate implementation
- ⟨Reflect⟩: Self-validation and improvement
"""

import logging
from typing import Dict, Any, Optional, List

from cerebro.llm.ollama_backend import OllamaBackend

logger = logging.getLogger(__name__)


class HybridBackend:
    """
    Hybrid backend combining DeepSeek's reliability with DeepAnalyze's agentic approach.
    
    Features:
    - Action token-based prompts (⟨Analyze⟩, ⟨Code⟩, ⟨Reflect⟩)
    - Reflection loops for self-improvement
    - Agentic, goal-oriented prompts
    - Fast inference with DeepSeek 16B
    """
    
    def __init__(
        self,
        model: str = "deepseek-coder-v2:16b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 300,
        enable_reflection: bool = True,
        max_reflection_iterations: int = 2
    ):
        """
        Initialize Hybrid backend.
        
        Args:
            model: DeepSeek model name
            base_url: Ollama server URL
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            timeout: Request timeout
            enable_reflection: Enable self-reflection loops
            max_reflection_iterations: Max iterations for reflection
        """
        self.ollama = OllamaBackend(
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
        self.enable_reflection = enable_reflection
        self.max_reflection_iterations = max_reflection_iterations
        
        logger.info(f"Hybrid backend initialized (DeepSeek + DeepAnalyze approach)")
        logger.info(f"  Model: {model}")
        logger.info(f"  Reflection: {'enabled' if enable_reflection else 'disabled'}")
    
    def reason(self, prompt: str, use_action_tokens: bool = True) -> str:
        """
        Generate reasoning using action tokens (DeepAnalyze style).
        
        Args:
            prompt: Task description
            use_action_tokens: Wrap with action tokens
            
        Returns:
            Generated response
        """
        if use_action_tokens:
            # Use DeepAnalyze-style action tokens with DeepSeek
            enhanced_prompt = f"""You are an expert data scientist. Use structured reasoning:

⟨Analyze⟩ First, analyze the task and plan your approach.

⟨Code⟩ Then, generate the implementation.

Task: {prompt}

Begin with ⟨Analyze⟩:
"""
            return self.ollama.chat(
                messages=[{"role": "user", "content": enhanced_prompt}]
            )
        else:
            return self.ollama.chat(
                messages=[{"role": "user", "content": prompt}]
            )
    
    def generate_code_with_reflection(
        self,
        task: str,
        data_info: Dict[str, Any],
        agent_name: str = "Agent"
    ) -> str:
        """
        Generate code with reflection loop (DeepAnalyze approach).
        
        Args:
            task: Task description
            data_info: Information about available data
            agent_name: Name of the agent for logging
            
        Returns:
            Generated Python code
        """
        # Step 1: Analyze + Code (agentic prompt)
        code = self._generate_initial_code(task, data_info, agent_name)
        
        # Step 2: Reflect + Improve (if enabled)
        if self.enable_reflection:
            code = self._reflect_and_improve(code, task, data_info, agent_name)
        
        return code
    
    def _generate_initial_code(
        self,
        task: str,
        data_info: Dict[str, Any],
        agent_name: str
    ) -> str:
        """
        Generate initial code using agentic, goal-oriented prompt.
        
        This uses DeepAnalyze's approach:
        - Goal-oriented (WHAT to achieve, not HOW)
        - Minimal constraints
        - Trust the model to make decisions
        """
        # Build minimal, agentic prompt
        media_channels = data_info.get('media_channels', [])
        kpi_col = data_info.get('kpi_col', 'target')
        date_col = data_info.get('date_col', 'date')
        
        # Show only first 3 channels for brevity
        channels_display = media_channels[:3]
        if len(media_channels) > 3:
            channels_display = f"{channels_display}... ({len(media_channels)} total)"
        
        prompt = f"""⟨Analyze⟩ You are a {agent_name} expert for Marketing Mix Models.

Task: {task}

Data context (already loaded in scope):
- data: DataFrame with {len(media_channels)} media channels
- media_channels: {channels_display}
- kpi_col: '{kpi_col}'
- date_col: '{date_col}'

⟨Analyze⟩ Plan your approach:
1. What transformations/models are best for this data?
2. What decisions do you need to make?
3. How will you validate quality?

⟨Code⟩ Generate production-ready Python code:
- Use the variables already defined (data, media_channels, kpi_col)
- Make autonomous decisions (choose methods, parameters)
- Store results in appropriate output variable
- Add minimal comments explaining key decisions

Generate clean, concise code (20-30 lines):
"""
        
        logger.info(f"[Hybrid/{agent_name}] Generating initial code with agentic prompt")
        response = self.ollama.chat(
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract code from response
        code = self._extract_code(response)
        return code
    
    def _reflect_and_improve(
        self,
        code: str,
        task: str,
        data_info: Dict[str, Any],
        agent_name: str
    ) -> str:
        """
        Reflect on generated code and improve it (DeepAnalyze approach).
        
        Args:
            code: Initially generated code
            task: Original task
            data_info: Data context
            agent_name: Agent name
            
        Returns:
            Improved code
        """
        for iteration in range(self.max_reflection_iterations):
            logger.info(f"[Hybrid/{agent_name}] Reflection iteration {iteration + 1}/{self.max_reflection_iterations}")
            
            # Ask model to reflect
            reflection_prompt = f"""⟨Reflect⟩ Review this code for a {agent_name} task.

Original task: {task}

Generated code:
```python
{code}
```

⟨Analyze⟩ Self-check:
1. Does it use the correct variable names (data, media_channels, kpi_col)?
2. Are there any hardcoded values that should be dynamic?
3. Does it make reasonable autonomous decisions?
4. Is the output variable correctly named?
5. Any potential errors (undefined variables, wrong column names)?

⟨Code⟩ If improvements needed, generate IMPROVED code. Otherwise, respond with "APPROVED".

Response:
"""
            
            reflection = self.ollama.chat(
                messages=[{"role": "user", "content": reflection_prompt}]
            )
            
            # Check if approved
            if "APPROVED" in reflection.upper() and "```" not in reflection:
                logger.info(f"[Hybrid/{agent_name}] Code approved after {iteration + 1} reflection(s)")
                break
            
            # Extract improved code
            improved = self._extract_code(reflection)
            if improved and improved != code and len(improved) > 10:
                logger.info(f"[Hybrid/{agent_name}] Code improved in reflection {iteration + 1}")
                code = improved
            else:
                logger.info(f"[Hybrid/{agent_name}] No significant improvements, stopping reflection")
                break
        
        return code
    
    def _extract_code(self, response: str) -> str:
        """
        Extract Python code from LLM response.
        
        Args:
            response: Full LLM response
            
        Returns:
            Extracted code
        """
        # Look for code blocks
        if "```python" in response:
            parts = response.split("```python")
            if len(parts) > 1:
                code_part = parts[1].split("```")[0]
                return code_part.strip()
        
        # Look for plain code blocks
        if "```" in response:
            parts = response.split("```")
            if len(parts) >= 3:
                code_part = parts[1]
                # Skip if it's markdown/other
                if not code_part.strip().startswith(('json', 'bash', 'yaml', 'sql')):
                    return code_part.strip()
        
        # Look for ⟨Code⟩ section
        if "⟨Code⟩" in response or "<Code>" in response:
            # Find the code section
            for marker in ["⟨Code⟩", "<Code>", "⟨code⟩", "<code>"]:
                if marker in response:
                    code_section = response.split(marker)[1]
                    # Extract until next tag or end
                    for end_marker in ["⟨", "<", "```"]:
                        if end_marker in code_section:
                            code_section = code_section.split(end_marker)[0]
                            break
                    
                    # Clean up
                    code = code_section.strip()
                    if code and len(code) > 10:
                        return code
        
        # If no markers, try to find Python code patterns
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # Detect code start
            if any(keyword in line for keyword in ['import ', 'def ', 'class ', 'for ', 'if ', '= ']):
                in_code = True
            
            if in_code:
                # Stop at natural breaks
                if line.strip() and not line[0].isspace() and line.strip()[0] not in '#=<⟨':
                    if any(word in line.lower() for word in ['analyze', 'note:', 'explanation', 'summary']):
                        break
                
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        # Fallback: return full response
        logger.warning(f"[Hybrid] Could not extract clean code, using full response")
        return response.strip()


def get_hybrid_backend(**kwargs) -> HybridBackend:
    """Create Hybrid backend with default settings."""
    return HybridBackend(**kwargs)

