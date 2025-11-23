"""
🎯 Modeling Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Autonomously generates Bayesian MMM model code based on spec.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import logging
import yaml
from cerebro.agents.base_agent import BaseAgent
from cerebro.llm import AutoBackend, RAGBackend
from cerebro.spec.schema import MMMSpec

logger = logging.getLogger(__name__)


class ModelingAgent(BaseAgent):
    """
    Writes complete Bayesian MMM model code.
    
    Generates:
    - Transformation functions (adstock, saturation)
    - NumPyro/PyMC model function
    - Training function (SVI/NUTS)
    """
    
    def __init__(self, llm: AutoBackend, use_rag: bool = True):
        super().__init__(llm, "ModelingAgent")
        self.rag = RAGBackend() if use_rag else None
    
    def generate_model_code(self, spec: MMMSpec, data_path: str = None) -> str:
        """Generate complete modeling code autonomously"""
        logger.info("🎯 ModelingAgent generating Bayesian model code...")
        
        rag_context = self._get_modeling_examples(spec) if self.rag else ""
        
        spec_yaml = yaml.dump(spec.model_dump(exclude_none=True), sort_keys=False)
        
        prompt = f"""You are an expert Bayesian modeler. Write COMPREHENSIVE, DETAILED, PRODUCTION-GRADE NumPyro SVI Marketing Mix Model code.

SPEC:
```yaml
{spec_yaml}
```

{rag_context}

⚠️  CRITICAL API CORRECTNESS RULES - FOLLOW EXACTLY:
1. Use ONLY the API patterns shown in the examples above
2. Check JAX/NumPyro examples for correct shape handling
3. Use autoguide.AutoNormal, NOT AutoNormal
4. Use Trace_ELBO(), NOT numpyro.elbo.JointELBO()
5. Scalar parameters MUST remain scalars in adstock/saturation
6. Use jnp.power(scalar, array) for broadcasting (NOT array ** array)
7. Import pattern: from numpyro.infer import SVI, Trace_ELBO, autoguide
8. Optimizer: from numpyro.optim import Adam

CRITICAL: This must be DETAILED, production-quality code (300-400 lines minimum).

Write COMPREHENSIVE modeling code with multiple functions:

## 1. TRANSFORMATION FUNCTIONS (100-120 lines total)

{self._list_required_transforms(spec)}

For EACH transformation, write a DETAILED function with:
- Full docstring explaining the transformation
- Type hints for parameters
- JAX/NumPyro compatible operations (jnp.*)
- Edge case handling (zeros, negative values, normalization)
- Efficient vectorized implementation
- Parameter validation
- Comments explaining the math

Example transformations needed:
- geometric_adstock(x, alpha, max_lag): decay = alpha^lag
- weibull_adstock(x, shape, scale, max_lag): PDF-based decay
- delayed_adstock(x, delay, decay, max_lag): delayed peak
- hill_saturation(x, alpha, beta): S-curve with inflection
- logistic_saturation(x, lambda_): asymptotic saturation

## 2. COMPREHENSIVE MODEL FUNCTION (120-150 lines)

Write `mmm_model(data_dict, n_obs)` that:

1. BASELINE & INTERCEPT (15 lines):
   - Sample baseline from {spec.priors.baseline if hasattr(spec, 'priors') else 'normal(0,1)'}
   - Initialize mu as jnp.ones(n_obs) * baseline
   - Add seasonal components (if specified)

2. CHANNEL EFFECTS (80-100 lines - detailed for EACH of {len(spec.channels)} channels):
   For EACH channel, generate separate code blocks:
   
   Example for channel '{spec.channels[0].name if spec.channels else 'channel_1'}':
   ```
   # Channel: {spec.channels[0].name if spec.channels else 'channel_1'}
   raw_channel_1 = data_dict['{spec.channels[0].name if spec.channels else 'channel_1'}']
   
   # Adstock transformation
   adstock_params_1 = numpyro.sample('adstock_param_1', dist.Beta(2, 5))
   adstocked_1 = geometric_adstock(raw_channel_1, adstock_params_1, max_lag=...)
   
   # Saturation transformation  
   sat_alpha_1 = numpyro.sample('sat_alpha_1', dist.Gamma(3, 1))
   sat_beta_1 = numpyro.sample('sat_beta_1', dist.Gamma(3, 1))
   saturated_1 = hill_saturation(adstocked_1, sat_alpha_1, sat_beta_1)
   
   # Channel coefficient
   coef_1 = numpyro.sample('coef_channel_1', dist.HalfNormal(0.5))
   mu = mu + coef_1 * saturated_1
   ```
   
   Repeat this pattern for ALL channels with their specific transformations from the spec.

3. CONTROL EFFECTS (20-25 lines):
   For EACH control variable:
   - Sample coefficient from appropriate prior
   - Add to mu
   - Include comments

4. LIKELIHOOD & RESIDUALS (10 lines):
   - Sample sigma from {spec.priors.sigma if hasattr(spec, 'priors') else 'half_normal(100)'}
   - Define Normal likelihood
   - Return mu for predictions

## 3. SVI TRAINING FUNCTION (60-80 lines)

Write `run_svi(data_dict, outcome, n_obs, num_steps=10000, learning_rate=0.01)` that:

1. Setup (20 lines):
   - Create AutoNormal guide
   - Initialize Adam optimizer with learning rate schedule
   - Create SVI object
   - Setup RNG key
   - Initialize SVI state

2. Training Loop (30 lines):
   - Run for num_steps iterations
   - Track ELBO loss
   - Log progress every 500 steps with loss, time
   - Implement early stopping if loss plateaus
   - Gradient clipping for stability

3. Posterior Sampling (20 lines):
   - Get posterior samples (1000 samples)
   - Compute predictive samples
   - Calculate convergence diagnostics
   - Return samples, losses, svi, svi_state

## 4. HELPER FUNCTIONS (40 lines)

Write additional utility functions:
- `extract_channel_contributions(posterior_samples, data_dict)`: Calculate contribution per channel
- `calculate_roi(posterior_samples, data_dict, outcome)`: Compute ROI metrics
- `get_model_summary(posterior_samples)`: Print parameter summaries

OUTPUT ONLY PYTHON CODE. No markdown, no explanations, just Python and # comments.

TARGET: 300-400 lines of detailed, production-grade Python code.

{rag_context}

Write COMPREHENSIVE NumPyro SVI MMM functions for {len(spec.channels)} channels:

# delayed_adstock(x, params, max_lag) - delayed peak, full docstring, JAX ops
# geometric_adstock(x, params, max_lag) - geometric decay, full docstring, JAX ops  
# weibull_adstock(x, shape, scale, max_lag) - Weibull PDF decay, full docstring
# hill_saturation(x, alpha, beta) - S-curve, full docstring, parameter validation
# logistic_saturation(x, lambda_) - asymptotic saturation, full docstring

# mmm_model(data_dict, n_obs) - DETAILED NumPyro model:
#   - Baseline with priors
#   - For EACH of {len(spec.channels)} channels: adstock + saturation + coefficient sampling
#   - Control variables
#   - Likelihood
#   - Full docstring, comments for each step

# run_svi(data_dict, outcome, n_obs, num_steps=50000, lr=0.01) - COMPREHENSIVE training:
#   - AutoNormal guide setup
#   - Adam optimizer with learning rate schedule
#   - Training loop with ELBO tracking, progress logging every 500 steps
#   - Early stopping logic
#   - Posterior sampling (1000 samples)
#   - Full docstring, error handling

# extract_channel_contributions(posterior_samples, data_dict) - helper function
# calculate_roi(posterior_samples, data_dict, outcome) - ROI metrics  
# get_model_summary(posterior_samples) - parameter summaries

Include: type hints, comprehensive docstrings, error handling, progress prints, comments explaining math.

START WITH: import jax
END WITH: return posterior_samples, losses

Output 300-400 lines of valid Python code only.Output ONLY valid Python code with # comments.
- NO markdown code fences (``` or ```python)
- NO explanatory paragraphs or prose
- NO example usage sections
- NO numbered lists or bullet points
- Every line must be executable Python or a # comment
- Do NOT explain what the code does - just write the code Every line must be valid Python or a # comment."""

        # Stream the code generation
        print("\n" + "="*80)
        print("🎯 BAYESIAN MODEL CODE (streaming):")
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
        
        logger.info(f"✓ Generated {len(code.splitlines())} lines of model code")
        return code
    
    def _list_required_transforms(self, spec: MMMSpec) -> str:
        """List all transformations needed"""
        adstock_types = set()
        saturation_types = set()
        
        for ch in spec.channels:
            if ch.transform and ch.transform.adstock:
                adstock_types.add(ch.transform.adstock.type)
            if ch.transform and ch.transform.saturation:
                saturation_types.add(ch.transform.saturation.type)
        
        result = "Required transformation functions:\n\n"
        
        if adstock_types:
            result += "ADSTOCK FUNCTIONS:\n"
            for t in sorted(adstock_types):
                result += f"- {t}_adstock(x, params, max_lag)\n"
        
        if saturation_types:
            result += "\nSATURATION FUNCTIONS:\n"
            for t in sorted(saturation_types):
                result += f"- {t}_saturation(x, params)\n"
        
        return result
    
    def _get_modeling_examples(self, spec: MMMSpec) -> str:
        """Get modeling examples from RAG - prioritize API correctness"""
        backend = spec.inference.backend
        
        queries = [
            # API-specific queries for correct syntax
            "numpyro SVI autoguide AutoNormal Trace_ELBO example",
            "jax jnp.power broadcasting shape operations example",
            "numpyro.sample dist.Beta dist.Gamma distribution example",
            f"{backend} bayesian marketing mix model implementation",
            "adstock transformation geometric weibull delayed implementation",
            "saturation transformation hill logistic curve",
            "numpyro inference adam optimizer training loop",
        ]
        
        examples = []
        for q in queries:
            try:
                results = self.rag.search(q, n_results=3)  # More examples for API patterns
                if results:
                    examples.extend(results)
            except:
                continue
        
        if not examples:
            return ""
        
        return f"""
🔥 PRODUCTION MMM & API EXAMPLES - USE THESE EXACT PATTERNS:
{'═' * 80}
{chr(10).join([f"📚 Example {i+1}:{chr(10)}{ex[:2000]}{chr(10)}{'─' * 80}" for i, ex in enumerate(examples[:8])])}
{'═' * 80}
⚠️  COPY the API patterns above EXACTLY - especially imports, SVI setup, and shape operations!
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

