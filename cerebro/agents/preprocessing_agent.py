"""
🔧 Preprocessing Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Autonomously generates data preprocessing code for MMM.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import logging
from cerebro.agents.base_agent import BaseAgent
from cerebro.llm import AutoBackend, RAGBackend
from cerebro.spec.schema import MMMSpec

logger = logging.getLogger(__name__)


class PreprocessingAgent(BaseAgent):
    """
    Writes data preprocessing code.
    
    Generates functions for:
    - Missing value imputation
    - Outlier detection and capping
    - Feature scaling
    - Time feature engineering
    - Data quality checks
    """
    
    def __init__(self, llm: AutoBackend, use_rag: bool = True):
        super().__init__(llm, "PreprocessingAgent")
        self.rag = RAGBackend() if use_rag else None
    
    def generate_preprocessing_code(self, spec: MMMSpec, data_path: str = None) -> str:
        """Generate preprocessing code autonomously"""
        logger.info("🔧 PreprocessingAgent generating preprocessing code...")
        
        rag_context = self._get_preprocessing_examples() if self.rag else ""
        
        prompt = f"""You are an expert ML engineer. Write COMPREHENSIVE, DETAILED, PRODUCTION-GRADE preprocessing code for Marketing Mix Modeling.

{rag_context}

CRITICAL: This must be DETAILED code (150-200 lines minimum).

Write comprehensive preprocessing functions with:

1. COMPREHENSIVE MISSING VALUE HANDLING (40 lines):
   - Detect missing patterns
   - Forward fill for time series continuity
   - Backward fill for edge cases
   - Interpolation (linear, polynomial) for gaps
   - Mean/median imputation as fallback
   - Report missing value statistics before/after
   - Handle zero vs NaN distinction

2. ADVANCED OUTLIER TREATMENT (40 lines):
   - Z-score method (threshold 3)
   - IQR method (1.5*IQR)
   - Winsorization (cap at 1st/99th percentile)
   - Separate treatment for channels vs outcome
   - Log transformation for right-skewed channels
   - Report outlier counts and treatment applied

3. FEATURE ENGINEERING (50 lines):
   - Create lag features (1-4 weeks) for channels
   - Rolling averages (7, 14, 28 day windows)
   - Rate of change features
   - Seasonal indicators (month, quarter, holiday)
   - Day of week, week of year
   - Year-over-year growth rates
   - Moving standard deviations

4. SCALING & NORMALIZATION (30 lines):
   - StandardScaler for continuous variables
   - MinMaxScaler for bounded features
   - Log transformation for skewed distributions
   - Save scaler objects for inverse transform
   - Handle zero/negative values before log

5. DATA VALIDATION (30 lines):
   - Check for data leakage
   - Ensure no infinities or NaNs remain
   - Validate date continuity
   - Check for constant columns
   - VIF for multicollinearity (warn if >10)
   - Ensure sufficient observations (10x features)
   - Generate preprocessing report dict

Include:
- Full docstrings with parameter descriptions
- Type hints (from typing import)
- Rich print statements with emojis and progress
- Comprehensive try/except error handling
- Logging for debugging
- Helper functions for reusability
- Comments explaining each transformation

CRITICAL: Output ONLY valid Python code with # comments.
- NO markdown code fences (``` or ```python)
- NO explanatory paragraphs or prose
- NO example usage sections
- NO numbered lists or bullet points
- Every line must be executable Python or a # comment
- Do NOT explain what the code does - just write the code Every line must be valid Python or a # comment."""

        # Stream the code generation
        print("\n" + "="*80)
        print("🔧 PREPROCESSING CODE (streaming):")
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
        
        logger.info(f"✓ Generated {len(code.splitlines())} lines of preprocessing code")
        return code
    
    def _get_preprocessing_examples(self) -> str:
        """Get preprocessing examples from RAG"""
        queries = [
            "data preprocessing missing values outlier detection pandas",
            "feature scaling StandardScaler normalization",
            "time series feature engineering date extraction"
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
PRODUCTION PREPROCESSING EXAMPLES:
{'─' * 80}
{chr(10).join([f"Example {i+1}:{chr(10)}{ex[:1000]}{chr(10)}" for i, ex in enumerate(examples[:3])])}
{'─' * 80}
"""
    
    def _clean_code(self, code: str) -> str:
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

