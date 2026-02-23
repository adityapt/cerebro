"""
 Optimization Agent

Autonomously generates budget optimization code.

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
    
    def _get_reasoning_prompt(self, spec: MMMSpec, context: dict) -> str:
        """Generate COT reasoning prompt for optimization"""
        return f"""
You are optimizing budget allocation. REASON about the optimization strategy.

SPEC: Channels={len(spec.channels)}, Outcome={spec.outcome}

CRITICAL REASONING QUESTIONS:

1) OBJECTIVE: What are we optimizing?
   - Maximize ROI (Return on Investment)
   - Maximize incremental outcome for given budget
   - Find optimal allocation across channels

2) INPUT FORMAT: Where does data come from?
   - JSON manifest with paths to: params_path, predictions_path, metadata_path
   - params CSV: parameter samples (columns = param names)
   - Need: posterior_samples (coefficients), channel_names, budget

3) PARAMETER MAPPING - CRITICAL:
   - Modeling module outputs generic parameter names: 'coefs_0', 'coefs_1', 'coefs_2', etc.
   - These correspond to channels by INDEX, not by name
   - posterior_samples keys are 'coefs_X', NOT 'impressions_Channel_01'
   - To get coefficient for channel i, use: posterior_samples['coefs_{{i}}']
   - DO NOT try to access posterior_samples[channel_name] - it will cause KeyError!
   - Instead: enumerate channels and use index to build key

4) OPTIMIZATION METHOD: How to allocate budget?
   - Greedy by ROI: Sort channels by coefficient, allocate to highest first
   - Scipy optimization: Use scipy.optimize.minimize with constraints
   - Marginal analysis: Allocate based on marginal ROI

OUTPUT JSON: {{"objective": "maximize ROI", "input_format": "JSON manifest with params CSV", "param_access_pattern": "posterior_samples['coefs_0'] not posterior_samples[channel_name]", "optimization_method": "greedy by coefficient"}}

JSON only.
"""
    
    def generate_optimization_code(self, spec: MMMSpec, data_path: str = None) -> str:
        """Generate optimization code autonomously with COT reasoning"""
        logger.info(" OptimizationAgent generating optimization code...")
        
        # COT REASONING
        reasoning = self._reason_about_task(spec, {'data_path': data_path})
        reasoning_context = f"\n[COT Analysis: {reasoning.get('objective', 'N/A')}]\n\n" if reasoning else ""
        
        rag_context = self._get_optimization_examples() if self.rag else ""
        
        # Extract spec details
        channel_names = [ch.name for ch in spec.channels]
        
        prompt = f"""{reasoning_context}You are an expert in marketing optimization. Write a Python function for budget allocation.

CRITICAL: OUTPUT ONLY PYTHON CODE. NO explanations, NO markdown fences, NO "Sure", NO "Here is".
Start directly with 'import' statements or function definitions.

{rag_context}

CRITICAL - USE THESE EXACT VALUES FROM SPEC:
- OUTCOME: '{spec.outcome}'
- CHANNEL_NAMES: {channel_names}
- NUMBER OF CHANNELS: {len(channel_names)}

Write helper function `optimize_budget(posterior_samples, channel_names, current_spend, total_budget)` that:

1. Extract channel coefficients from posterior (mean) - CRITICAL:
   - Use enumerate(channel_names) to get index i for each channel
   - Access coefficient using: posterior_samples[f'coefs_{{i}}'].mean()
   - DO NOT access posterior_samples[channel_name] - it will fail!
   - Build dict mapping channel_name -> coefficient_value
   
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
- Docstring and comments

CRITICAL - YOU MUST ALSO INCLUDE THIS ENTRY POINT FUNCTION:

def run_optimization(results_path: str) -> dict:
    '''Main entry point for optimization module.
    
    Args:
        results_path (str): Path to JSON manifest (module_3_results.json)
        
    Returns:
        dict: Optimization results with optimal allocation
    '''
    import json
    import pandas as pd
    
    # Load manifest
    with open(results_path, 'r') as f:
        manifest = json.load(f)
    
    # Load data from CSV files
    params_df = pd.read_csv(manifest['params_path'])
    
    # Convert params back to dict (samples are rows, params are columns)
    posterior_samples = {{col: params_df[col].values for col in params_df.columns}}
    
    # CRITICAL NOTE: posterior_samples has keys like 'coefs_0', 'coefs_1', etc.
    # NOT channel names like 'impressions_Channel_01'!
    # Inside optimize_budget, use enumerate(channel_names) to map:
    #   for i, channel in enumerate(channel_names):
    #       coef = posterior_samples[f'coefs_{{i}}'].mean()
    
    # Use actual channel names from spec (passed as {channel_names})
    channel_names = {channel_names}
    n_channels = len(channel_names)
    
    # Default total budget (can be adjusted)
    total_budget = 100000.0
    
    # Assume uniform current spend as baseline
    current_spend = {{ch: total_budget / n_channels for ch in channel_names}}
    
    opt_results = optimize_budget(posterior_samples, channel_names, current_spend, total_budget)
    print(f"Optimization completed. Expected lift: {{opt_results.get('expected_lift', 'N/A')}}")
    
    return opt_results

CRITICAL REMINDERS:
- OUTPUT ONLY PYTHON CODE (no conversational preamble)
- Access posterior_samples using 'coefs_0', 'coefs_1', etc. via enumerate(channel_names)
- DO NOT access posterior_samples[channel_name] directly - it will cause KeyError!
- MUST include the run_optimization(results_path: str) -> dict function as the main entry point"""

        # Collect streaming code
        print("\n" + "="*80)
        print(" OPTIMIZATION CODE (streaming):")
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
        
        logger.info(f"[OK] Generated {len(code.splitlines())} lines of optimization code")
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
{'' * 80}
{chr(10).join([ex[:1000] for ex in examples[:2]])}
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

