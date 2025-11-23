"""
📊 Data Exploration Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Autonomously generates comprehensive data exploration code for MMM.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import logging
from cerebro.agents.base_agent import BaseAgent
from cerebro.llm import AutoBackend, RAGBackend
from cerebro.spec.schema import MMMSpec

logger = logging.getLogger(__name__)


class DataExplorationAgent(BaseAgent):
    """
    Writes comprehensive data exploration code.
    
    Generates functions for:
    - Time series analysis
    - Autocorrelation checks
    - Channel spending patterns
    - Cross-correlations
    - Missing data analysis
    - Outlier detection
    """
    
    def __init__(self, llm: AutoBackend, use_rag: bool = True):
        super().__init__(llm, "DataExplorationAgent")
        self.rag = RAGBackend() if use_rag else None
    
    def generate_eda_code(self, spec: MMMSpec, data_path: str = None) -> str:
        """Generate data exploration code autonomously"""
        logger.info("🔍 DataExplorationAgent generating EDA code...")
        
        # Get RAG examples
        rag_context = self._get_eda_examples() if self.rag else ""
        
        # Build prompt  
        data_info = f"\nDATA PATH: {data_path}" if data_path else ""
        prompt = f"""OUTPUT ONLY PYTHON CODE. No markdown, no explanations, no prose. Just Python code and # comments.

TARGET: 250-300 lines of detailed, production-grade Python code.

{rag_context}

# Data: {spec.outcome} with {len(spec.channels)} channels
{data_info}

Write COMPREHENSIVE EDA functions for MMM:

# load_data(file_path) - DETAILED: Load CSV, validate ALL columns, check dtypes, memory usage, duplicates, missing value report, date parsing
# descriptive_statistics(df, channels) - DETAILED: For EACH channel - mean, median, std, min, max, percentiles (5,25,50,75,95,99), skewness, kurtosis, coefficient of variation, spend concentration
# time_series_analysis(df, outcome, channels) - DETAILED: Trend decomposition, seasonality (weekly/monthly), rolling stats (7/30 day), ACF/PACF for outcome AND all channels, stationarity tests (ADF, KPSS), changepoint detection
# correlation_analysis(df, channels, outcome) - DETAILED: Pearson/Spearman/Kendall, VIF calculation for multicollinearity, condition number, cross-correlation with lags
# outlier_detection(df, channels) - DETAILED: Z-score (threshold 3), IQR method, Isolation Forest, outlier counts per channel, visualization

Each function: full docstring, type hints, try/except error handling, logging statements, matplotlib/seaborn visualizations, progress prints with emojis, comments explaining statistical methods.

START WITH: import pandas as pd
END WITH: return results

Output 250-300 lines of valid Python code only. Every line must be Python or # comment."""

        # Agent writes code with STREAMING
        print("\n" + "="*80)
        print("📊 DATA EXPLORATION CODE (streaming):")
        print("="*80 + "\n")
        
        full_code = ""
        for token in self.llm.reason(prompt, stream=True):
            print(token, end="", flush=True)
            full_code += token
        
        print("\n\n" + "="*80)
        code = full_code
        # Aggressive cleanup - remove all non-Python lines
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip markdown fences
            if stripped.startswith('```'):
                continue
            # Skip prose (sentences ending with period, no Python keywords)
            if stripped and not any([
                line.startswith(' '),
                line.startswith('\t'),
                line.startswith('#'),
                line.startswith('def '),
                line.startswith('class '),
                line.startswith('import '),
                line.startswith('from '),
                line.startswith('@'),
                '=' in line,
                stripped.endswith(':'),
                any(kw in line for kw in ['if ', 'for ', 'while ', 'try:', 'except', 'return ', 'yield ', 'raise '])
            ]) and (
                stripped[0].isdigit() or
                '**' in stripped or
                (stripped.endswith('.') and len(stripped.split()) > 5)
            ):
                continue
            cleaned_lines.append(line)
        code = '\n'.join(cleaned_lines)

        
        # Clean up

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
        
        logger.info(f"✓ Generated {len(code.splitlines())} lines of EDA code")
        return code
    
    def _get_eda_examples(self) -> str:
        """Get EDA examples from RAG"""
        queries = [
            "pandas exploratory data analysis time series autocorrelation",
            "marketing data profiling channel spending patterns",
            "outlier detection missing data analysis statistics"
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
PRODUCTION EDA EXAMPLES:
{'─' * 80}
{chr(10).join([f"Example {i+1}:{chr(10)}{ex[:1000]}{chr(10)}" for i, ex in enumerate(examples[:3])])}
{'─' * 80}
"""
    
    def _clean_code(self, code: str) -> str:
        """Remove markdown and artifacts"""
        code = code.strip()
        if code.startswith('```python'):
            code = code[len('```python'):].strip()
        if code.startswith('```'):
            code = code[3:].strip()
        if code.endswith('```'):
            code = code[:-3].strip()
        for token in ['<|im_end|>', '<|endoftext|>']:
            code = code.replace(token, '')
        return code.strip()
