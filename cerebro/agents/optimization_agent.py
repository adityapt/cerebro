"""
💰 Optimization Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Autonomously generates budget optimization code.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import logging
from cerebro.agents.base_agent import BaseAgent
from cerebro.llm import AutoBackend, RAGBackend
from cerebro.spec.schema import MMMSpec

logger = logging.getLogger(__name__)


class OptimizationAgent(BaseAgent):
    """Writes budget optimization and ROI analysis code"""
    
    def __init__(self, llm: AutoBackend, use_rag: bool = True):
        super().__init__(llm, "OptimizationAgent")
        self.rag = RAGBackend() if use_rag else None
    
    def generate_optimization_code(self, spec: MMMSpec, data_path: str = None) -> str:
        """Generate optimization code autonomously"""
        logger.info("💰 OptimizationAgent generating optimization code...")
        
        rag_context = self._get_optimization_examples() if self.rag else ""
        
        prompt = f"""You are an expert in marketing optimization. Write a Python function for budget allocation.

{rag_context}

Write `optimize_budget(posterior_samples, channel_names, current_spend, total_budget)` that:

1. Extract channel coefficients from posterior (mean)
2. Calculate current ROI for each channel (coef / spend)
3. Sort channels by ROI
4. Allocate budget starting with highest ROI channels
5. Calculate expected lift vs current allocation

Args:
- posterior_samples: dict with parameter samples
- channel_names: list of channel names  
- current_spend: dict of current spend by channel
- total_budget: total budget to allocate

Returns:
- optimal_allocation: dict with optimal spend
- expected_lift: expected incremental outcome

Include:
- Detailed print statements showing current vs optimal
- Greedy allocation algorithm
- Expected impact calculation
- Docstring and comments"""

        code = self.llm.reason(prompt, stream=True)

        # AGGRESSIVE cleanup
        lines = code.split('\n')
        cleaned = []
        for line in lines:
            s = line.strip()
            # Skip markdown fences
            if s in ['```', '```python', '```py'] or s.startswith('```'):
                continue
            # Skip LLM tokens
            if '<|im_end|>' in line or '<|endoftext|>' in line:
                continue
            # Skip numbered prose
            if s and not line[0].isspace() and s[0].isdigit() and '. ' in s[:4]:
                continue
            # Skip "This function..." or "The function..."
            if s.startswith(('This function', 'The function', 'This code', 'Example usage:')):
                continue
            cleaned.append(line)
        code = '\n'.join(cleaned)
        code = self._clean_code(code)
        
        logger.info(f"✓ Generated {len(code.splitlines())} lines of optimization code")
        return code
    
    def _get_optimization_examples(self) -> str:
        queries = [
            "budget optimization ROI allocation algorithm",
            "marginal return on investment calculation",
            "greedy allocation optimal spend"
        ]
        
        examples = []
        for q in queries:
            try:
                results = self.rag.search(q, n_results=1)
                if results:
                    examples.extend(results)
            except:
                continue
        
        if not examples:
            return ""
        
        return f"""
PRODUCTION OPTIMIZATION EXAMPLES:
{'─' * 80}
{chr(10).join([ex[:1000] for ex in examples[:2]])}
{'─' * 80}
"""
    
    def _clean_code(self, code: str) -> str:
        code = code.strip()
        for prefix in ['```python', '```']:
            if code.startswith(prefix):
                code = code[len(prefix):].strip()
        if code.endswith('```'):
            code = code[:-3].strip()
        for token in ['<|im_end|>', '<|endoftext|>']:
            code = code.replace(token, '')
        return code.strip()

