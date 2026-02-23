"""
 Diagnostics Agent

Autonomously generates model diagnostics code.

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
    
    def _get_reasoning_prompt(self, spec: MMMSpec, context: dict) -> str:
        """Generate COT reasoning prompt for diagnostics"""
        upstream_fields = context.get('upstream_fields', [])
        has_losses = context.get('has_losses', False)
        upstream_schema = context.get('upstream_schema', {})
        
        return f"""
You are diagnosing an MMM model. REASON about what diagnostics are needed.

SPEC: Backend={spec.inference.backend}, Channels={len(spec.channels)}

ACTUAL UPSTREAM DATA AVAILABLE (from modeling module):
- Fields in manifest: {upstream_fields}
- Has 'losses' field: {has_losses}
- Full schema: {upstream_schema}

CRITICAL REASONING QUESTIONS:

1) WHAT TO CHECK: What diagnostics validate the model?
   - Convergence (loss curve, R-hat, ESS) - ONLY if 'losses' field exists!
   - Fit quality (R², MAPE, residuals) - ALWAYS available
   - Parameter estimates (means, stds, credible intervals) - ALWAYS available

2) INPUT FORMAT: Where does data come from?
   - JSON manifest with paths to: predictions_path, params_path, metadata_path
   - predictions CSV: 'actual', 'predicted' columns
   - params CSV: parameter samples (columns = param names)
   - ACTUAL fields available: {upstream_fields}

3) PARAMETER MAPPING - CRITICAL:
   - Modeling module outputs generic parameter names: 'coefs_0', 'coefs_1', 'coefs_2', etc.
   - These correspond to channels by INDEX, not by name
   - posterior_samples keys are 'coefs_X', NOT 'impressions_Channel_01'
   - To access parameters, use: posterior_samples['coefs_0'], posterior_samples['coefs_1'], etc.
   - DO NOT try to access posterior_samples['impressions_Channel_01'] - it will cause KeyError!

4) OUTPUT FORMAT: What should diagnostics return?
   - Dict with: r2, mape, residual_stats, parameter_summary, convergence_metrics

5) REQUIRED IMPORTS - CRITICAL:
   - For polyfit, mean, std: numpy
   - For R², MAPE: sklearn.metrics (r2_score, mean_absolute_percentage_error)
   - For R-hat, ESS: arviz (az.from_dict, az.rhat, az.ess) - OPTIONAL (wrap in try/except)
   - DO NOT import scipy.signal, scipy.ndimage, or any unused libraries!
   - Only import what is actually USED in the code
   - ARVIZ WARNING: Some environments have arviz/scipy version conflicts, so make arviz usage OPTIONAL with try/except

OUTPUT JSON: {{"diagnostics_needed": ["r2", "mape", "residuals", "posterior_stats", "convergence"], "input_format": "JSON manifest with CSV paths", "param_access_pattern": "posterior_samples['coefs_0'] not posterior_samples[channel_name]", "required_imports": ["numpy", "sklearn.metrics", "arviz"], "forbidden_imports": ["scipy.signal", "scipy.ndimage"], "output_format": "dict with metrics"}}

JSON only.
"""
    
    def generate_diagnostics_code(self, spec: MMMSpec, data_path: str = None, upstream_output: dict = None) -> str:
        """Generate diagnostics code autonomously with COT reasoning
        
        Args:
            spec: MMM specification
            data_path: Path to data file
            upstream_output: Schema/manifest from modeling module (CRITICAL for data-aware generation)
        """
        logger.info(" DiagnosticsAgent generating diagnostics code...")
        
        # COT REASONING with actual upstream data
        context = {'data_path': data_path}
        if upstream_output:
            context['upstream_fields'] = list(upstream_output.keys())
            context['has_losses'] = 'losses' in upstream_output
            context['has_channel_names'] = 'channel_names' in upstream_output
            context['upstream_schema'] = upstream_output
        
        reasoning = self._reason_about_task(spec, context)
        reasoning_context = f"\n[COT Analysis: {reasoning.get('diagnostics_needed', [])}]\n\n" if reasoning else ""
        
        rag_context = self._get_diagnostics_examples() if self.rag else ""
        
        # Extract spec details
        channel_names = [ch.name for ch in spec.channels]
        
        prompt = f"""{reasoning_context}You are an expert in Bayesian model diagnostics. Write a Python function for comprehensive model validation.

CRITICAL: OUTPUT ONLY PYTHON CODE. NO explanations, NO markdown fences, NO "Sure", NO "Here is".
Start directly with 'import' statements or function definitions.

{rag_context}

CRITICAL - USE THESE EXACT VALUES FROM SPEC:
- OUTCOME: '{spec.outcome}'
- CHANNEL_NAMES: {channel_names}
- BACKEND: '{spec.inference.backend}'

CRITICAL - IMPORTS (ONLY import what you actually USE):
- numpy (for np.polyfit, np.mean, np.std)
- sklearn.metrics (for r2_score, mean_absolute_percentage_error)
- pandas, json (for data loading in run_diagnostics)
- arviz as az (OPTIONAL - some environments have arviz/scipy conflicts):
  * Import arviz ONLY inside the try/except block where it's used
  * DO NOT import arviz at the top of the file
  * Example: try: import arviz as az; diagnostics['rhat'] = az.rhat(...) except: pass
- DO NOT import scipy.signal, scipy.ndimage, or any other unused libraries!

Write helper function `diagnose_model(posterior_samples, actual, predicted, r2, svi_state=None)` that:

1. Convergence Analysis (OPTIONAL - only if svi_state has losses):
   - Check if svi_state exists and has 'losses' attribute
   - If yes: Check loss curve trend (polyfit slope), print final loss
   - If no: Skip convergence analysis

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

CRITICAL - YOU MUST ALSO INCLUDE THIS ENTRY POINT FUNCTION:

def run_diagnostics(results_path: str) -> dict:
    '''Main entry point for diagnostics module.
    
    Args:
        results_path (str): Path to JSON manifest (module_3_results.json)
        
    Returns:
        dict: Diagnostics results
    '''
    import json
    import pandas as pd
    
    # Load manifest
    with open(results_path, 'r') as f:
        manifest = json.load(f)
    
    # Load data from CSV files
    predictions_df = pd.read_csv(manifest['predictions_path'])
    params_df = pd.read_csv(manifest['params_path'])
    metadata_df = pd.read_csv(manifest['metadata_path'])
    
    # Extract data
    actual = predictions_df['actual'].values
    predicted = predictions_df['predicted'].values
    r2 = float(metadata_df[metadata_df['metric'] == 'r2']['value'].values[0])
    
    # Convert params back to dict (samples are rows, params are columns)
    posterior_samples = {{col: params_df[col].values for col in params_df.columns}}
    
    # CRITICAL NOTE: posterior_samples has keys like 'coefs_0', 'coefs_1', etc.
    # NOT channel names like 'impressions_Channel_01'!
    # Access parameters by their generic names: posterior_samples['coefs_0']
    
    diag_results = diagnose_model(posterior_samples, actual, predicted, r2)
    print(f"Diagnostics completed: {{diag_results.get('summary', 'Done')}}")
    
    return diag_results

CRITICAL REMINDERS:
- OUTPUT ONLY PYTHON CODE (no conversational preamble)
- Access posterior_samples using 'coefs_0', 'coefs_1', etc. (NOT channel names)
- Include error handling, docstring, comments
- MUST include the run_diagnostics(results_path: str) -> dict function as the main entry point

FEW-SHOT EXAMPLES:

EXAMPLE 1 - GOOD (COMPLETE MODULE WITH BOTH FUNCTIONS):
```python
import numpy as np
import pandas as pd
import json
from sklearn.metrics import r2_score, mean_absolute_percentage_error

def diagnose_model(posterior_samples, actual, predicted, r2, svi_state=None):
    diagnostics = {{}}
    
    # Convergence Analysis (optional)
    if svi_state and hasattr(svi_state, 'losses'):
        losses = svi_state.losses
        slope, _ = np.polyfit(range(len(losses)), losses, 1)
        diagnostics['convergence_slope'] = slope
        diagnostics['final_loss'] = losses[-1]
    
    # Posterior Predictive Checks
    diagnostics['r2'] = r2
    diagnostics['mape'] = mean_absolute_percentage_error(actual, predicted)
    residuals = actual - predicted
    diagnostics['residuals_mean'] = np.mean(residuals)
    diagnostics['residuals_std'] = np.std(residuals)
    
    # Parameter Summary
    param_summary = {{}}
    for i in range(min(10, len(posterior_samples))):
        key = f'coefs_{{i}}'
        param_summary[key] = {{
            'mean': np.mean(posterior_samples[key]),
            'std': np.std(posterior_samples[key])
        }}
    diagnostics['param_summary'] = param_summary
    diagnostics['total_params'] = len(posterior_samples)
    
    # ArviZ diagnostics (optional - may fail in some environments)
    try:
        import arviz as az
        inference_data = az.from_dict(posterior=posterior_samples)
        diagnostics['rhat'] = az.rhat(inference_data).to_dict()
        diagnostics['ess'] = az.ess(inference_data).to_dict()
    except (ImportError, Exception) as e:
        diagnostics['arviz_error'] = str(e)
    
    return diagnostics

def run_diagnostics(results_path: str) -> dict:
    '''Main entry point for diagnostics module.'''
    with open(results_path, 'r') as f:
        manifest = json.load(f)
    
    predictions_df = pd.read_csv(manifest['predictions_path'])
    params_df = pd.read_csv(manifest['params_path'])
    metadata_df = pd.read_csv(manifest['metadata_path'])
    
    actual = predictions_df['actual'].values
    predicted = predictions_df['predicted'].values
    r2 = float(metadata_df[metadata_df['metric'] == 'r2']['value'].values[0])
    
    posterior_samples = {{col: params_df[col].values for col in params_df.columns}}
    
    diag_results = diagnose_model(posterior_samples, actual, predicted, r2)
    print(f"Diagnostics completed")
    
    return diag_results
```
^ CORRECT: Shows COMPLETE pattern with both diagnose_model AND run_diagnostics entry point

EXAMPLE 2 - BAD (INCOMPLETE - MISSING ENTRY POINT):
```python
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_percentage_error

def diagnose_model(posterior_samples, actual, predicted, r2):
    diagnostics = {{}}
    diagnostics['r2'] = r2
    diagnostics['mape'] = mean_absolute_percentage_error(actual, predicted)
    return diagnostics

# ← WRONG: Missing run_diagnostics function! Module won't execute!
```
^ REJECTED: Missing the run_diagnostics entry point function - module incomplete!"""

        # Collect streaming code
        print("\n" + "="*80)
        print(" DIAGNOSTICS CODE (streaming):")
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
        
        logger.info(f"[OK] Generated {len(code.splitlines())} lines of diagnostics code")
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
{'' * 80}
{chr(10).join([ex[:1000] for ex in examples[:3]])}
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

