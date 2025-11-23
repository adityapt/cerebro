"""
📈 Visualization Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Autonomously generates visualization code.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import logging
from cerebro.agents.base_agent import BaseAgent
from cerebro.llm import AutoBackend, RAGBackend
from cerebro.spec.schema import MMMSpec

logger = logging.getLogger(__name__)


class VisualizationAgent(BaseAgent):
    """Writes comprehensive visualization code"""
    
    def __init__(self, llm: AutoBackend, use_rag: bool = True):
        super().__init__(llm, "VisualizationAgent")
        self.rag = RAGBackend() if use_rag else None
    
    def generate_visualization_code(self, spec: MMMSpec, data_path: str = None) -> str:
        """Generate visualization code autonomously"""
        logger.info("📈 VisualizationAgent generating visualization code...")
        
        rag_context = self._get_visualization_examples() if self.rag else ""
        
        prompt = f"""You are an expert in data visualization. Write a Python function for MMM visualizations using matplotlib and seaborn.

{rag_context}

Write `visualize_results(data, outcome, predictions, posterior_samples, channel_cols, date_col=None)` that creates:

1. Actual vs Predicted Plot:
   - 2 subplots: time series and residual plot
   - If date_col, use it as x-axis
   - Save as 'mmm_predictions.png'

2. Channel Contributions:
   - Calculate contribution = coef × channel_data for each channel
   - Create horizontal bar chart (top 10 channels)
   - Green for positive, red for negative
   - Save as 'mmm_contributions.png'

3. Posterior Distributions:
   - Histograms for first 6 parameters
   - 2x3 subplot grid
   - Save as 'mmm_posteriors.png'

Return dict with contributions and plot filenames.

Use:
- matplotlib.pyplot for plotting
- figsize=(12, 6) or larger
- dpi=300 for high quality
- plt.tight_layout()
- Proper labels and titles

Include docstring and comments."""

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
        
        logger.info(f"✓ Generated {len(code.splitlines())} lines of visualization code")
        return code
    
    def _get_visualization_examples(self) -> str:
        queries = [
            "matplotlib seaborn time series actual predicted plot",
            "channel contribution waterfall bar chart visualization",
            "posterior distribution histogram subplot"
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
PRODUCTION VISUALIZATION EXAMPLES:
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

