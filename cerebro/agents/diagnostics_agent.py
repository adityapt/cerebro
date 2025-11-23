"""
🔍 Diagnostics Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Autonomously generates model diagnostics code.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import logging
from cerebro.agents.base_agent import BaseAgent
from cerebro.llm import AutoBackend, RAGBackend
from cerebro.spec.schema import MMMSpec

logger = logging.getLogger(__name__)


class DiagnosticsAgent(BaseAgent):
    """Writes model diagnostics and validation code"""
    
    def __init__(self, llm: AutoBackend, use_rag: bool = True):
        super().__init__(llm, "DiagnosticsAgent")
        self.rag = RAGBackend() if use_rag else None
    
    def generate_diagnostics_code(self, spec: MMMSpec, data_path: str = None) -> str:
        """Generate diagnostics code autonomously"""
        logger.info("🔍 DiagnosticsAgent generating diagnostics code...")
        
        rag_context = self._get_diagnostics_examples() if self.rag else ""
        
        prompt = f"""You are an expert in Bayesian model diagnostics. Write a Python function for comprehensive model validation.

{rag_context}

Write `diagnose_model(posterior_samples, losses, data, outcome, svi_state=None)` that:

1. Convergence Analysis:
   - Check loss curve trend (polyfit slope)
   - Print final loss

2. Posterior Predictive Checks:
   - Calculate predictions (mean of posterior mu or obs)
   - Compute R² (using sklearn.metrics.r2_score)
   - Compute MAPE (using sklearn.metrics.mean_absolute_percentage_error)
   - Analyze residuals (mean, std)

3. Parameter Summary:
   - Print mean ± std for first 10 parameters
   - Count total parameters

4. ArviZ Diagnostics (if available):
   - Convert to inference_data
   - Check R-hat (< 1.01 is good)
   - Check ESS (> 400 recommended)

Return diagnostics dict with all results.

Include error handling, docstring, comments."""

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
        
        logger.info(f"✓ Generated {len(code.splitlines())} lines of diagnostics code")
        return code
    
    def _get_diagnostics_examples(self) -> str:
        queries = [
            "bayesian model diagnostics convergence R-hat ESS",
            "posterior predictive checks sklearn metrics",
            "residual analysis validation"
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
PRODUCTION DIAGNOSTICS EXAMPLES:
{'─' * 80}
{chr(10).join([ex[:1000] for ex in examples[:3]])}
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

