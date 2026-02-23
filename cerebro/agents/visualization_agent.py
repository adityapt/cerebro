"""
 Visualization Agent

Autonomously generates visualization code.

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
    
    def _get_reasoning_prompt(self, spec: MMMSpec, context: dict) -> str:
        """Generate COT reasoning prompt for visualization"""
        return f"""
You are creating MMM visualizations. REASON CAREFULLY about data requirements.

SPEC: Backend={spec.inference.backend}, Channels={len(spec.channels)}, Outcome={spec.outcome}

CRITICAL REASONING QUESTIONS (ANSWER ALL):

1) KEY PLOTS: What visualizations are needed?
   - Actual vs Predicted plot?
   - Channel contribution plot?
   - Posterior distribution plot?

2) DATA REQUIREMENTS PER PLOT:
   - Actual vs Predicted: needs actual values + predictions (from predictions_path CSV)
   - Channel Contributions: needs ORIGINAL channel data + coefficients (from data_path CSV + params_path CSV)
   - Posterior Distributions: needs MCMC samples (from params_path CSV)

3) DATA SOURCES - WHICH CSV FILES TO LOAD:
   - manifest['predictions_path']: Contains actual vs predicted values
   - manifest['params_path']: Contains MCMC parameter samples (coefficients)
   - manifest['data_path']: Contains ORIGINAL channel data (REQUIRED for contributions!)
   - manifest['param_mapping']: Maps parameter names (coefs_0, coefs_1) to channel names (impressions_Channel_01, etc.)

4) PARAMETER NAME MAPPING - CRITICAL FOR ACCESSING COEFFICIENTS:
   - MCMC outputs generic parameter names: 'coefs_0', 'coefs_1', 'coefs_2', etc.
   - Channel names are: 'impressions_Channel_01', 'impressions_Channel_02', etc.
   - posterior_samples keys are 'coefs_X', NOT channel names!
   - To get coefficient for channel[i], use: posterior_samples[f'coefs_{{i}}']
   - OR use manifest['param_mapping'] to map channel names to parameter names

5) CRITICAL: Channel contribution calculation
   - Formula: contribution = coefficient × channel_data
   - You need BOTH: coefficients (from params) AND channel_data (from ORIGINAL data)
   - Must load original data CSV, not just predictions!
   - When iterating channels, use enumerate(channel_cols) to get index for coefs_X

6) FUNCTION SIGNATURE:
   - visualize_results(data, outcome, predictions, posterior_samples, channel_cols)
   - 'data' parameter = original DataFrame (NOT None!)
   - Must pass original data to calculate contributions
   - When calculating contributions, iterate with enumerate to map channel index to coefs_X

OUTPUT JSON: {{"key_plots": ["actual_vs_predicted", "channel_contributions", "posterior_distributions"], "data_requirements": {{"predictions": "predictions_path", "params": "params_path", "original_data": "data_path (REQUIRED for contributions!)"}}, "must_load_original_data": true, "pass_data_not_none": true, "parameter_mapping": "Use enumerate(channel_cols) to map index to coefs_X"}}

JSON only. THINK: To calculate channel contributions, you MUST load original data!
"""
    
    def generate_visualization_code(self, spec: MMMSpec, data_path: str = None) -> str:
        """Generate visualization code autonomously with COT reasoning"""
        logger.info(" VisualizationAgent generating visualization code...")
        
        # COT REASONING
        reasoning = self._reason_about_task(spec, {'data_path': data_path})
        reasoning_context = f"\n[COT Analysis: Plots={reasoning.get('key_plots', [])}]\n\n" if reasoning else ""
        
        rag_context = self._get_visualization_examples() if self.rag else ""
        
        # Extract spec details
        channel_names = [ch.name for ch in spec.channels]
        
        prompt = f"""{reasoning_context}You are a Python code generator for MMM visualizations.

CRITICAL OUTPUT RULES:
1. OUTPUT ONLY VALID PYTHON CODE
2. NO explanations, NO markdown fences (```), NO conversational text
3. NO "Sure, here's...", NO "Below is...", NO "Here is...", etc.
4. Start directly with 'import' statements
5. If you include ANY non-code text, the system will FAIL

FEW-SHOT EXAMPLES (LEARN THE CORRECT FORMAT):

EXAMPLE 1 - GOOD:
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(actual, predicted):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    ax1.scatter(actual, predicted, alpha=0.5)
    ax1.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual', fontsize=12)
    ax1.set_ylabel('Predicted', fontsize=12)
    ax1.set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    residuals = actual - predicted
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', lw=2)
    ax2.set_xlabel('Residuals', fontsize=12)
    ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    return 'predictions.png'

EXAMPLE 2 - GOOD (CRITICAL: Parameter Mapping with enumerate!):
def calculate_contributions(data, posterior_samples, channel_cols):
    '''Calculate channel contributions using CORRECT parameter mapping.
    
    CRITICAL PATTERN: Use enumerate(channel_cols) to map index to coefs_X!
    posterior_samples has keys like 'coefs_0', 'coefs_1', NOT channel names!
    '''
    contributions = {{}}
    
    #  CORRECT: Use enumerate to map channel index to parameter name
    for idx, channel in enumerate(channel_cols):
        coef_key = f'coefs_{{idx}}'  # Map channel index to parameter name
        coef = posterior_samples[coef_key].mean()  # Access using coefs_X
        channel_data = data[channel].values
        contributions[channel] = coef * channel_data.sum()
    
    # Sort by absolute contribution
    sorted_items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    channels, values = zip(*sorted_items) if sorted_items else ([], [])
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.barh(range(len(channels)), values, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels, fontsize=10)
    ax.set_xlabel('Contribution', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Channel Contributions', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('contributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    return 'contributions.png'

EXAMPLE 3 - GOOD (Helper Function WITH Docstring - CRITICAL PATTERN):
def visualize_results(data, outcome, predictions, posterior_samples, channel_cols):
    '''Create comprehensive MMM visualizations.
    
    Args:
        data: DataFrame with input data
        outcome: Actual outcome values
        predictions: Predicted outcome values
        posterior_samples: Dictionary of MCMC samples
        channel_cols: List of channel column names
    
    Returns:
        dict: Dictionary with plot filenames and contributions
    '''
    # CRITICAL: Note closing ''' BEFORE any executable code!
    # Create actual vs predicted plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    ax1.plot(outcome, label='Actual', color='blue')
    ax1.plot(predictions, label='Predicted', color='red', linestyle='--')
    ax1.set_title('Actual vs Predicted', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    residuals = outcome - predictions
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax2.set_title('Residuals', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {{'plots': ['predictions.png']}}

EXAMPLE 4 - GOOD:
def plot_posteriors(posterior_samples):
    params = list(posterior_samples.keys())[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=300)
    axes = axes.flatten()
    
    for idx, param in enumerate(params):
        samples = np.array(posterior_samples[param]).flatten()
        axes[idx].hist(samples, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].axvline(samples.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {{samples.mean():.3f}}')
        axes[idx].set_title(param, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Value', fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('posteriors.png', dpi=300, bbox_inches='tight')
    plt.close()
    return 'posteriors.png'

EXAMPLE 5 - GOOD (Entry Point with Data Loading - CRITICAL PATTERN):
def run_visualization(results_path: str) -> dict:
    '''Load results and create visualizations.
    
    Args:
        results_path: Path to JSON manifest
    
    Returns:
        dict with plot filenames
    '''
    import json
    import pandas as pd
    
    # Load manifest
    with open(results_path, 'r') as f:
        manifest = json.load(f)
    
    # CRITICAL: Load ALL required CSV files
    predictions_df = pd.read_csv(manifest['predictions_path'])
    params_df = pd.read_csv(manifest['params_path'])
    original_df = pd.read_csv(manifest['data_path'])  # -> MUST load original data!
    
    # Extract data
    actual = predictions_df['actual'].values
    predicted = predictions_df['predicted'].values
    posterior_samples = {{col: params_df[col].values for col in params_df.columns}}
    
    # Get channel names
    channel_cols = [col for col in original_df.columns if 'Channel' in col or 'channel' in col]
    
    # CRITICAL: Pass original_df, NOT None!
    plots = plot_results(original_df, actual, predicted, posterior_samples, channel_cols)
    
    return {{'plots': plots}}

EXAMPLE 6 - GOOD (Longer Docstring Shows Pattern):
def run_diagnostics(results_path: str) -> dict:
    '''Run model diagnostics including convergence checks.
    
    This function performs statistical validation of MCMC results
    including Rhat statistics, effective sample size, and
    posterior predictive checks.
    
    Args:
        results_path (str): Path to pickled model results
    
    Returns:
        dict: Diagnostic metrics and plot filenames
    
    Raises:
        FileNotFoundError: If results file missing
    '''  # -> THREE QUOTES TO CLOSE - ALWAYS!
    import pickle
    import numpy as np
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    diagnostics = {{'rhat': 1.01, 'n_eff': 500}}
    return diagnostics

EXAMPLE 7 - GOOD (Another Entry Point):
def run_optimization(results_path: str) -> dict:
    '''Optimize budget allocation across marketing channels.
    
    Args:
        results_path: Path to model results
    
    Returns:
        dict with optimal allocations
    '''  # -> ALWAYS CLOSE BEFORE IMPORTS!
    import scipy.optimize as opt
    import numpy as np
    
    # Budget optimization logic
    optimal = {{'channel_1': 1000, 'channel_2': 2000}}
    return optimal

EXAMPLE 8 - GOOD (Third Complete Example):
def run_preprocessing(data_path: str) -> str:
    '''Clean and prepare marketing data for modeling.
    
    Args:
        data_path: Input CSV file path
    
    Returns:
        str: Path to preprocessed CSV file
    '''  # �� DOCSTRING CLOSED HERE!
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv(data_path)
    df_clean = df.dropna()
    output = data_path.replace('.csv', '_clean.csv')
    df_clean.to_csv(output, index=False)
    return output

EXAMPLE 9 - BAD (WRONG FORMAT):
Sure, here's the visualization code you requested:

import matplotlib.pyplot as plt
...

^ REJECTED: Contains conversational text "Sure, here's..."

EXAMPLE 10 - BAD (WRONG FORMAT):
```python
import matplotlib.pyplot as plt
...
```

^ REJECTED: Contains markdown fences

EXAMPLE 11 - BAD (WRONG DOCSTRING - CRITICAL):
def run_viz(path: str) -> dict:
    '''Load results
    
    Args:
        path: file path
    import pickle  #  WRONG: Docstring not closed!
    ...

^ REJECTED: Docstring not properly closed with ''' - Code is inside docstring!

EXAMPLE 12 - BAD (UNCLOSED DOCSTRING WITH CODE):
def visualize_data(data, outcome, predictions):
    '''Create visualization plots.
    
    Args:
        data: Input dataframe
        outcome: Actual values
    # Create plot                     #  WRONG: Still inside docstring!
    fig, ax = plt.subplots()          #  WRONG: Code inside docstring!
    '''                                #  WRONG: Closing too late!
    ...

^ REJECTED: Docstring must close BEFORE any code. Move ''' to line after 'outcome: Actual values'

EXAMPLE 13 - BAD (WRONG PARAMETER MAPPING - CRITICAL ERROR):
def calculate_contributions_wrong(data, posterior_samples, channel_cols):
    '''This will FAIL with KeyError!'''
    contributions = {{}}
    
    #  WRONG: Accessing posterior_samples with channel name directly
    for channel in channel_cols:
        coef = posterior_samples[channel].mean()  # KeyError: 'impressions_Channel_01'!
        channel_data = data[channel].values
        contributions[channel] = coef * channel_data.sum()
    
    return contributions

^ REJECTED: posterior_samples has keys 'coefs_0', 'coefs_1', etc., NOT channel names!
  Use enumerate(channel_cols) to get index, then access posterior_samples[f'coefs_{{idx}}']

EXAMPLE 14 - BAD (PASSING data=None - CRITICAL ERROR):
def run_visualization(results_path: str) -> dict:
    '''Create visualizations.'''
    import json
    import pandas as pd
    
    with open(results_path, 'r') as f:
        manifest = json.load(f)
    
    predictions_df = pd.read_csv(manifest['predictions_path'])
    params_df = pd.read_csv(manifest['params_path'])
    #  WRONG: Forgot to load original data!
    
    actual = predictions_df['actual'].values
    predicted = predictions_df['predicted'].values
    posterior_samples = {{col: params_df[col].values for col in params_df.columns}}
    
    #  CRITICAL ERROR: Passing data=None will fail when calculating contributions!
    plots = plot_results(None, actual, predicted, posterior_samples, channel_cols)
    #                    ^^^^ WRONG: Must pass original DataFrame, not None!
    
    return {{'plots': plots}}

^ REJECTED: Must load original data from manifest['data_path'] and pass it, NOT None!

{rag_context}

NOW GENERATE YOUR CODE FOLLOWING THE GOOD EXAMPLES:

CRITICAL - USE THESE EXACT VALUES FROM SPEC:
- OUTCOME: '{spec.outcome}'
- CHANNEL_NAMES: {channel_names}
- DATE_COLUMN: '{spec.date_column}'
- BACKEND: '{spec.inference.backend}'

Write helper function `visualize_results(data, outcome, predictions, posterior_samples, channel_cols, date_col=None)` that creates:

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

Include docstring and comments.

CRITICAL - YOU MUST ALSO INCLUDE THIS ENTRY POINT FUNCTION:

def run_visualization(results_path: str) -> dict:
    '''Main entry point for visualization module.
    
    Args:
        results_path (str): Path to JSON manifest (module_3_results.json)
        
    Returns:
        dict: Visualization results with plot filenames
    '''
    import json
    import pandas as pd
    
    # Load manifest
    with open(results_path, 'r') as f:
        manifest = json.load(f)
    
    # CRITICAL: Load ALL required CSV files (including original data!)
    predictions_df = pd.read_csv(manifest['predictions_path'])
    params_df = pd.read_csv(manifest['params_path'])
    
    # MUST load original data for channel contribution calculations
    # Try data_path from manifest, fallback to examples/MMM Data.csv
    data_path = manifest.get('data_path', 'examples/MMM Data.csv')
    original_df = pd.read_csv(data_path)
    
    # Extract data
    actual = predictions_df['actual'].values
    predicted = predictions_df['predicted'].values
    
    # Convert params back to dict (samples are rows, params are columns)
    posterior_samples = {{col: params_df[col].values for col in params_df.columns}}
    
    # Use actual channel names from spec
    channel_cols = {channel_names}
    
    # CRITICAL: Pass original_df, NOT None!
    viz_results = visualize_results(original_df, actual, predicted, posterior_samples, channel_cols, date_col=None)
    print(f"Visualizations created: {{viz_results.get('plots', [])}}")
    
    return viz_results

MUST include the run_visualization(results_path: str) -> dict function as the main entry point.

CRITICAL DATA LOADING REQUIREMENTS:
1. MUST load original data: original_df = pd.read_csv(manifest.get('data_path', 'examples/MMM Data.csv'))
2. MUST pass original_df to visualize_results(), NOT None!
3. Channel contributions need: coefficient × channel_data (both required!)
4. If you pass data=None, contribution calculation will FAIL with TypeError!

FINAL REMINDER:
- Your FIRST line MUST be "import ..." (not conversational text)
- Your LAST line MUST be code (not explanations)
- NO markdown fences, NO "Sure", NO "Here is"
- ONLY Python code
- LOAD ORIGINAL DATA for contributions!"""

        # Collect streaming code
        print("\n" + "="*80)
        print(" VISUALIZATION CODE (streaming):")
        print("="*80 + "\n")
        
        full_code = ""
        for token in self.llm.reason(prompt, stream=True):
            print(token, end="", flush=True)
            full_code += token
        
        print("\n\n" + "="*80)
        code = full_code

        # AGGRESSIVE cleanup
        lines = code.split('\n')
        cleaned = []
        conversational_prefixes = (
            'Sure, here', 'Here is', 'Here are', 'Below is', 'Below are',
            'I will', "I'll", 'Let me', 'This is', 'Certainly',
            'This function', 'The function', 'This code', 'Example usage:',
            'To implement', 'First,', 'Now,', 'Next,', 'Finally,'
        )
        
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
            # Skip conversational prefixes (case-insensitive)
            if any(s.lower().startswith(prefix.lower()) for prefix in conversational_prefixes):
                continue
            # Skip lines with only punctuation or explanatory text
            if s and not any(c.isalnum() or c in '()[]{}' for c in s):
                continue
            cleaned.append(line)
        code = '\n'.join(cleaned)
        code = self._clean_code(code)
        
        # CRITICAL FIX: Auto-close unclosed docstrings (deterministic, not LLM-based)
        code = self._fix_unclosed_docstrings(code)
        
        logger.info(f"[OK] Generated {len(code.splitlines())} lines of visualization code")
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
                results = self.rag.retrieve(q, n_results=1)
                if results:
                    for r in results:
                        if 'output' in r:
                            examples.append(r['output'])
            except Exception as e:
                logger.debug(f"RAG query failed: {e}")
                continue
        
        if not examples:
            return ""
        
        return f"""
PRODUCTION VISUALIZATION EXAMPLES FROM RAG:
{'' * 80}
{chr(10).join([ex[:800] for ex in examples[:2]])}
{'' * 80}
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
    
    def _fix_unclosed_docstrings(self, code: str) -> str:
        """
        Deterministically fix unclosed docstrings using regex pattern matching.
        This is REGEX-BASED, not LLM-based, so it's 100% deterministic.
        """
        import re
        
        # Pattern: function def, then opening docstring, then content, but no closing before code
        # We'll use a simple state machine approach
        lines = code.split('\n')
        fixed_lines = []
        in_docstring = False
        docstring_quote = None
        docstring_indent = 0
        docstring_start_line = -1
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            
            # Case 1: Starting a docstring
            if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                # Check if it closes on the same line
                quote = stripped[:3]
                rest = stripped[3:]
                if quote in rest:  # Closes on same line
                    fixed_lines.append(line)
                    continue
                else:  # Multi-line docstring starts
                    in_docstring = True
                    docstring_quote = quote
                    docstring_indent = indent
                    docstring_start_line = i
                    fixed_lines.append(line)
                    continue
            
            # Case 2: Inside docstring
            if in_docstring:
                # Check if this line closes the docstring
                if docstring_quote in stripped:
                    in_docstring = False
                    fixed_lines.append(line)
                    continue
                
                # Check if we hit code without closing (import, from, def, =, etc.)
                code_patterns = [
                    r'^\s*import\s',           # import statements
                    r'^\s*from\s',             # from imports
                    r'^\s*def\s',              # function definitions
                    r'^\s*class\s',            # class definitions
                    r'^\s*if\s',               # conditionals
                    r'^\s*for\s',              # loops
                    r'^\s*while\s',            # while loops
                    r'^\s*return\s',           # return statements
                    r'^\s*with\s',             # context managers
                    r'^\s*\w+\s*=',            # simple assignment: x = ...
                    r'^\s*\w+[\[\(]',          # function calls: func(...) or indexing: arr[...]
                    r'^\s*[\w,\s\(\)]+\s*=',   # tuple/multiple assignment: x, y = ... or (a, b) = ...
                    r'^\s*fig\s*[,=]',         # matplotlib figure patterns
                    r'^\s*ax\d*\s*[,=\[]',     # matplotlib axes patterns: ax1, ax[0], ax =
                    r'^\s*plt\.',              # matplotlib pyplot calls
                    r'^\s*@\w+',               # decorators
                    r'^\s*try:',               # try blocks
                    r'^\s*except',             # except blocks
                    r'^\s*finally:',           # finally blocks
                    r'^\s*raise\s',            # raise statements
                    r'^\s*assert\s',           # assertions
                ]
                is_code = any(re.match(pattern, line) for pattern in code_patterns)
                
                # Check if this is a code comment (not a docstring comment)
                # Code comments typically start with # followed by action words
                is_code_comment = False
                if stripped.startswith('#') and not stripped.startswith('###'):
                    # Look for action words that indicate code sections
                    action_words = ['create', 'calculate', 'compute', 'load', 'save', 'plot', 
                                   'generate', 'initialize', 'setup', 'step', 'phase', 'actual vs',
                                   'extract', 'convert', 'build', 'construct']
                    comment_lower = stripped.lower()
                    if any(word in comment_lower for word in action_words):
                        is_code_comment = True
                
                if is_code or is_code_comment:
                    # Insert closing docstring before this line
                    closing_line = ' ' * docstring_indent + docstring_quote
                    fixed_lines.append(closing_line)
                    logger.info(f"Auto-fixed unclosed docstring at line {i+1} (started at line {docstring_start_line+1})")
                    in_docstring = False
                    docstring_quote = None
                    # Now add the current line
                    fixed_lines.append(line)
                    continue
                
                # Still in docstring, add line
                fixed_lines.append(line)
                continue
            
            # Case 3: Normal code outside docstring
            fixed_lines.append(line)
        
        # Final check: if still in docstring at end of file
        if in_docstring and docstring_quote:
            closing_line = ' ' * docstring_indent + docstring_quote
            fixed_lines.append(closing_line)
            logger.info(f"Auto-fixed unclosed docstring at end of file (started at line {docstring_start_line+1})")
        
        return '\n'.join(fixed_lines)

