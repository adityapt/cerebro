"""
 Modeling Agent

Autonomously generates Bayesian MMM model code based on spec.

"""
import logging
import yaml
from cerebro.agents.base_agent import BaseAgent
from cerebro.llm import AutoBackend, RAGBackend
from cerebro.spec.schema import MMMSpec

logger = logging.getLogger(__name__)

# Reference snippet for params CSV save (literal "param_name" so prompt f-string does not evaluate it)
_PARAMS_SAVE_REF = """
    params_data = {}
    for param_name, param_values in samples.items():
        if param_values.ndim == 1:
            params_data[param_name] = param_values
        else:
            for idx in np.ndindex(param_values.shape[1:]):
                params_data[f"{param_name}_{'_'.join(map(str, idx))}"] = param_values[(slice(None),) + idx]
    params_df = pd.DataFrame(params_data)
    params_df.to_csv('module_3_params.csv', index=False)
"""


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
    
    def _get_reasoning_prompt(self, spec: MMMSpec, context: dict) -> str:
        """Generate COT reasoning prompt for modeling"""
        adstock_types = list(set(ch.transform.adstock.type for ch in spec.channels))
        saturation_types = list(set(ch.transform.saturation.type for ch in spec.channels))
        
        return f"""
================================================================================
          BAYESIAN MARKETING MIX MODELING (MMM) - CORE CONCEPTS
================================================================================

You are building a BAYESIAN MMM - the gold standard for marketing attribution.

 MMM OBJECTIVE:
Decompose business outcome (sales, visits, etc.) into contributions from:
  1. Marketing channels (TV, digital, social, etc.)
  2. Control variables (pricing, promotions, seasonality)
  3. Baseline (organic demand)

Then answer: "What is the incremental impact of $1 spent on Channel A?"

================================================================================

 BAYESIAN MMM MATHEMATICAL STRUCTURE:

The model is:
  outcome_t = baseline + Σ(channel_effects) + Σ(control_effects) + noise

Where each channel effect is:
  channel_effect_t = coef_i * saturation(adstock(spend_i,t))

TRANSFORMATIONS (Critical for MMM!):

1. **ADSTOCK**: Marketing impact persists over time (ads today -> sales next 3 weeks)
   
   Types:
   - Geometric: x_transformed_t = x_t + λ * x_transformed_(t-1)
     * λ (lambda): decay rate [0, 1]
     * λ=0.5 means 50% of impact remains next week
   
   - Weibull: More flexible S-curve decay
   
   WHY: TV ad doesn't just impact today - it builds awareness over weeks!

2. **SATURATION**: Diminishing returns at high spend
   
   Types:
   - Hill (Michaelis-Menten): y = x^s / (k^s + x^s)
     * k: half-saturation point (where effect = 50% of max)
     * s: shape parameter (steepness)
   
   - Logistic: y = 1 / (1 + exp(-k*(x-L)))
   
   WHY: First $1000 of spend >> last $1000 of spend (law of diminishing returns)

3. **FULL TRANSFORMATION PIPELINE**:
   raw_spend -> adstock -> saturation -> coef * transformed_spend

================================================================================

BAYESIAN INFERENCE (Why Bayesian > Frequentist for MMM):

ADVANTAGES:
   Quantifies uncertainty (95% credible intervals on ROI)
   Incorporates prior knowledge (e.g., "TV coefficient should be positive")
   Handles multicollinearity better (via regularization priors)
   Produces full posterior distribution (not just point estimate)

PRIORS (Regularization + Domain Knowledge):

  - Baseline ~ Normal(mean_outcome, σ_baseline)
    * Center near observed mean outcome
    
  - Coefficients ~ HalfNormal(σ_coef) or Normal(0, σ_coef)
    * HalfNormal if we know channels have positive effect
    * Normal(0, σ) allows negative (e.g., cannibalization)
    * σ controls regularization strength
    
  - Sigma (noise) ~ HalfNormal(σ_prior)
    * If scaled data: use HalfNormal(0.1)
    * If raw data: use HalfNormal(σ_outcome / 2)
    
  - Transform params (λ, k, s) ~ Uniform or Beta
    * λ (adstock decay): Beta(2, 2) on [0, 1]
    * k (half-saturation): Uniform(0, 1) on scaled data
    * s (shape): Uniform(0.5, 3)

LIKELIHOOD:
  outcome ~ Normal(μ = predicted, σ = sigma)

INFERENCE BACKENDS:
  - MCMC (NUTS): Full posterior, slower, more accurate
    * Use when: < 50 channels, need uncertainty quantification
    * Samples: 1000-4000 (warmup + samples)
    
  - SVI (Variational Inference): Approximate posterior, faster
    * Use when: > 50 channels, speed needed
    * Steps: 10,000-50,000

================================================================================

 YOUR TASK: Build Bayesian MMM with NumPyro

SPEC INFO:
- Channels: {len(spec.channels)}
- Controls: {len(spec.controls) if spec.controls else 0}
- Adstock types: {adstock_types}
- Saturation types: {saturation_types}
- Inference: {spec.inference.backend}
- Samples: {spec.inference.num_samples if spec.inference.backend == 'numpyro_nuts' else 'N/A (SVI)'}
- Outcome: {spec.outcome}
- Advanced: time_varying_parameters={getattr(getattr(spec, 'advanced', None), 'time_varying_parameters', True) if getattr(spec, 'advanced', None) else True}, correlated_channel_priors={getattr(getattr(spec, 'advanced', None), 'correlated_channel_priors', True) if getattr(spec, 'advanced', None) else True}

��������������������������������������������������������������������������������������������������������������������������������������������������������������

 CRITICAL REASONING QUESTIONS FOR MMM:

1. TRANSFORMATION FUNCTIONS: What functions do I need to implement?
   
   Required: {adstock_types} + {saturation_types}
   
   Example:
   - geometric_adstock(x, lam): Apply decay recursively
   - hill_saturation(x, k, s): Apply Hill curve
   
    VECTORIZATION: Use JAX array operations (NOT loops!)
     * Sample all adstock λ params as array: lam ~ Beta(2, 2, shape=(n_channels,))
     * Apply adstock to ALL channels at once using vmap or jnp operations

2. DATA SCALING: Should I scale channels before transforms?
   
    YES! Scale to [0, 1] with MinMaxScaler
   
   WHY:
   - Makes priors meaningful (HalfNormal(0.1) on scaled data)
   - Prevents overflow in saturation curves
   - Hill function k parameter interpretable on [0, 1]
   
   WHEN: Before adstock/saturation, after loading data

3. VECTORIZATION: Use arrays or loops for channel parameters?
   
    ARRAYS! Sample all channel coefficients at once:
   
   CORRECT:
   coefs = numpyro.sample('coefs', dist.HalfNormal(0.1), shape=(n_channels,))
   channel_effects = coefs * transformed_channels  # vectorized!
   
   WRONG (DON'T DO THIS!):
   for i in range(n_channels):
       coef_i = numpyro.sample(f'coef_{{i}}', dist.HalfNormal(0.1))  # loops are slow!

4. MODEL STRUCTURE: What is the regression equation?
   
   predicted = baseline + sum(channel_effects) + sum(control_effects)
   
   Where:
   - baseline: intercept (scalar)
   - channel_effects: coefs[i] * saturation(adstock(channels[:, i]))
   - control_effects: control_coefs[j] * controls[:, j]
   
   Likelihood:
   numpyro.sample('obs', dist.Normal(predicted, sigma), obs=y_observed)

5. PRIORS: What priors are appropriate for scaled [0, 1] data?
   
   - baseline ~ Normal(outcome_mean, outcome_std / 2)
   - coefs ~ HalfNormal(0.1)  # scaled data �� small coefficients
   - sigma ~ HalfNormal(0.1)  # scaled data �� small noise
   - adstock_lambda ~ Beta(2, 2)  # decay rate [0, 1]
   - sat_k ~ Uniform(0.1, 0.9)  # half-saturation on [0, 1]
   - sat_s ~ Uniform(0.5, 3.0)  # shape parameter

6. EXECUTION ORDER: What is the correct pipeline?
   
   Step 1: Load preprocessed data
   Step 2: Extract channel columns, control columns, outcome
   Step 3: Scale channels to [0, 1] (MinMaxScaler)
   Step 4: Define transformation functions (adstock, saturation)
   Step 5: Define Bayesian model function
   Step 6: Run inference (MCMC or SVI)
   Step 7: Extract posterior samples
   Step 8: Generate predictions (on original scale!)
   Step 9: Calculate R² on original scale
   Step 10: Save results (predictions, params, metadata, manifest)

7. OUTPUT MANIFEST: What metadata should I save for downstream modules?
   
   CRITICAL for visualization/diagnostics:
   - data_path: path to original data
   - channel_names: list of channel column names
   - param_mapping: {{'coefs_0': 'channel_name_0', ...}}
   - predictions_path, params_path, metadata_path
   - r2: model fit metric

8. POTENTIAL ISSUES:
   
   - NaN in transforms (e.g., log of negative values)
   - Overflow in saturation (large k or s values)
   - R² calculated on wrong scale (should be UNSCALED outcome!)
   - Missing manifest fields (downstream modules fail!)
   - Not using vectorization (very slow for 13+ channels)

��������������������������������������������������������������������������������������������������������������������������������������������������������������

OUTPUT FORMAT: Return ONLY valid JSON:

{{
    "transformation_functions": ["geometric_adstock", "hill_saturation"],
    "scaling_strategy": "MinMaxScaler [0, 1] before adstock/saturation",
    "vectorization": "Sample all channel params as arrays (n_channels,), use vmap or array ops",
    "model_structure": "outcome = baseline + coefs @ transformed_channels + control_coefs @ controls + noise",
    "priors": {{
        "baseline": "Normal(mean_outcome, std_outcome / 2)",
        "coefs": "HalfNormal(0.1) for scaled data",
        "sigma": "HalfNormal(0.1) for scaled data",
        "adstock_lambda": "Beta(2, 2)",
        "sat_k": "Uniform(0.1, 0.9)",
        "sat_s": "Uniform(0.5, 3.0)"
    }},
    "execution_order": [
        "Load preprocessed data",
        "Extract channels, controls, outcome",
        "Scale channels to [0, 1]",
        "Define adstock/saturation functions",
        "Define Bayesian model",
        "Run MCMC/SVI inference",
        "Extract posterior samples",
        "Generate predictions (unscaled!)",
        "Calculate R² (unscaled!)",
        "Save results + manifest with channel_names and param_mapping"
    ],
    "potential_issues": [
        "NaN from transforms on edge cases",
        "R² on wrong scale",
        "Missing manifest fields",
        "Loops instead of vectorization",
        "Overflow in saturation curves"
    ],
    "validation": [
        "Check no NaN in predictions",
        "Verify R² in [0, 1] range",
        "Confirm manifest has data_path, channel_names, param_mapping",
        "Test transform functions on edge cases (0, 1, large values)"
    ]
}}

Think step-by-step about Bayesian MMM requirements. Output ONLY the JSON.
"""
    
    def generate_model_code(self, spec: MMMSpec, data_path: str = None) -> str:
        """Generate complete modeling code autonomously with COT reasoning"""
        logger.info(" ModelingAgent generating Bayesian model code...")
        
        # ========== PHASE 1: CHAIN OF THOUGHT REASONING ==========
        reasoning = self._reason_about_task(spec, {'data_path': data_path})
        
        if reasoning:
            logger.info(f"[COT] Transformations: {reasoning.get('transformation_functions', [])}")
            logger.info(f"[COT] Scaling: {reasoning.get('scaling_strategy', 'N/A')}")
            logger.info(f"[COT] Vectorization: {reasoning.get('vectorization', 'N/A')[:50]}...")
        
        spec_yaml = yaml.dump(spec.model_dump(exclude_none=True), sort_keys=False)
        
        # Use simple example-first prompt for MCMC
        if spec.inference.backend == "numpyro_nuts":
            return self._generate_mcmc_code_simple(spec, spec_yaml, reasoning)
        else:
            return self._generate_svi_code_simple(spec, spec_yaml, reasoning)
        
        prompt = f"""You are an expert Bayesian modeler. Write COMPREHENSIVE, DETAILED, PRODUCTION-GRADE NumPyro {backend_type} Marketing Mix Model code.

SPEC:
```yaml
{spec_yaml}
```

{rag_context}

[WARNING]  CRITICAL API CORRECTNESS RULES - FOLLOW EXACTLY:
1. Use ONLY the API patterns shown in the examples above
2. Check JAX/NumPyro examples for correct shape handling
3. Use autoguide.AutoNormal, NOT AutoNormal
4. Use Trace_ELBO(), NOT numpyro.elbo.JointELBO()
5. Scalar parameters MUST remain scalars in adstock/saturation
6. Use jnp.power(scalar, array) for broadcasting (NOT array ** array)
7. Import pattern: from numpyro.infer import SVI, Trace_ELBO, autoguide
8. Optimizer: from numpyro.optim import Adam

FEW-SHOT EXAMPLES FOR VECTORIZED MCMC:

Example 1: BAD (slow, channel-by-channel)
```python
# DON'T DO THIS - processes each channel individually
for i, channel_name in enumerate(channel_names):
    decay = numpyro.sample(f'decay_{{i}}', dist.Beta(3, 3))
    alpha = numpyro.sample(f'alpha_{{i}}', dist.LogNormal(0, 0.5))
    coef = numpyro.sample(f'coef_{{i}}', dist.HalfNormal(0.3))
    x = geometric_adstock(data_dict[channel_name], decay, 8)
    x = hill_saturation(x, alpha, 0.5)
    mu += coef * x
```

Example 2: GOOD (fast, vectorized with arrays)
```python
# DO THIS - processes all channels in parallel
channel_names = ['channel_1', 'channel_2', 'channel_3']  # All channel names
n_channels = len(channel_names)

# Stack all channel data into array (n_channels, n_obs)
channel_data = jnp.stack([data_dict[name] for name in channel_names])

# Sample ALL parameters as arrays
decays = numpyro.sample('decays', dist.Beta(3, 3).expand([n_channels]))
alphas = numpyro.sample('alphas', dist.LogNormal(0, 0.5).expand([n_channels]))
coefs = numpyro.sample('coefs', dist.HalfNormal(0.3 / jnp.sqrt(n_channels)).expand([n_channels]))

# Vectorized transformation function
def transform_channel(ch_data, decay, alpha, coef):
    x = geometric_adstock(ch_data, decay, 8)
    x = hill_saturation(x, alpha, 0.5)
    return coef * x

# Apply to ALL channels using vmap (parallel)
# PRODUCTION INPUT CONTRACT: channel_data (n_channels, n_obs); decays, alphas, coefs must be (n_channels,) so vmap batches over axis 0. Do NOT pass (n_channels, n_obs) as coefs.
contributions = jax.vmap(transform_channel)(channel_data, decays, alphas, coefs)
mu += jnp.sum(contributions, axis=0)
```

Example 3: VECTORIZED controls
```python
# Stack all controls (n_controls, n_obs)
control_data = jnp.stack([data_dict[name] for name in control_names])
n_controls = len(control_names)

# Sample ALL control coefficients at once
control_coefs = numpyro.sample('control_coefs', dist.Normal(0, 0.3).expand([n_controls]))

# Vectorized: multiply and sum
mu += jnp.sum(control_data * control_coefs[:, None], axis=0)
```

ALWAYS use vectorized approach (Example 2 & 3) for MCMC - it's 10-100x faster!

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
- exponential_adstock(x, decay, max_lag): exponential decay
- delayed_adstock(x, delay, decay, max_lag): delayed peak
- hill_saturation(x, alpha, beta): S-curve with inflection
- logistic_saturation(x, lambda_): asymptotic saturation

## 2. COMPREHENSIVE MODEL FUNCTION (120-150 lines)

Write `mmm_model(data_dict, n_obs)` that:

1. BASELINE & INTERCEPT (15 lines):
   - Sample baseline from {spec.priors.baseline if hasattr(spec, 'priors') else 'normal(0,1)'}
   - Initialize mu as jnp.ones(n_obs) * baseline
   - Add seasonal components (if specified)

2. CHANNEL EFFECTS (60-80 lines - VECTORIZED for ALL {len(spec.channels)} channels):
   
   CRITICAL FOR MCMC SPEED: Use VECTORIZED operations, NOT channel-by-channel loops!
   
   Vectorized approach:
   ```
   # Stack ALL {len(spec.channels)} channels into single array (n_channels, n_obs)
   channel_data = jnp.stack([data_dict[ch_name] for ch_name in channel_names])
   n_channels = channel_data.shape[0]
   
   # Sample ALL channel coefficients at once (vectorized)
   channel_coefs = numpyro.sample('channel_coefs', 
       dist.HalfNormal(0.3 / jnp.sqrt(n_channels)).expand([n_channels]))
   
   # Sample ALL adstock parameters at once  
   adstock_alphas = numpyro.sample('adstock_alphas', 
       dist.Beta(2, 5).expand([n_channels]))
   
   # Sample ALL saturation parameters at once
   sat_alphas = numpyro.sample('sat_alphas', dist.Gamma(3, 1).expand([n_channels]))
   sat_betas = numpyro.sample('sat_betas', dist.Gamma(3, 1).expand([n_channels]))
   
   # Define vectorized transformation function
   def transform_single_channel(ch_data, alpha, sat_a, sat_b, coef):
       adstocked = geometric_adstock(ch_data, alpha, max_lag=8)
       saturated = hill_saturation(adstocked, sat_a, sat_b)
       return coef * saturated
   
   # Apply to ALL channels using jax.vmap (parallel vectorization)
   channel_contributions = jax.vmap(transform_single_channel)(
       channel_data, adstock_alphas, sat_alphas, sat_betas, channel_coefs
   )
   
   # Sum all channel effects
   mu = mu + jnp.sum(channel_contributions, axis=0)
   ```
   
   Use ARRAYS and jax.vmap - this is 10-100x faster than channel-by-channel sampling!

3. CONTROL EFFECTS (20-25 lines):
   For EACH control variable:
   - Sample coefficient from appropriate prior
   - Add to mu
   - Include comments

4. LIKELIHOOD & RESIDUALS (10 lines):
   - Sample sigma from {spec.priors.sigma if hasattr(spec, 'priors') else 'half_normal(100)'}
   - Define Normal likelihood
   - Return mu for predictions

{training_section}

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
# exponential_adstock(x, decay, max_lag) - exponential decay, full docstring, JAX ops
# hill_saturation(x, alpha, beta) - S-curve, full docstring, parameter validation
# logistic_saturation(x, lambda_) - asymptotic saturation, full docstring

# mmm_model(data_dict, n_obs) - DETAILED NumPyro model:
#   - Baseline with priors
#   - VECTORIZED processing of ALL {len(spec.channels)} channels using jax.vmap (arrays, not loops!)
#   - Control variables
#   - Likelihood
#   - Full docstring, comments for each step

# MAIN ENTRY POINT (EXACT SIGNATURE AND STRUCTURE REQUIRED):
# {function_example}
#   STRUCTURE: Load data -> Setup MCMC -> Run mcmc.run() -> Get samples -> Compute predictions -> Return dict
#   This MUST contain the FULL implementation as shown in the training guidance below
#   DO NOT create wrapper functions - put all MCMC logic directly in this function!

# extract_channel_contributions(posterior_samples, data_dict) - helper function
# calculate_roi(posterior_samples, data_dict, outcome) - ROI metrics  
# get_model_summary(posterior_samples) - parameter summaries

Include: type hints, comprehensive docstrings, error handling, progress prints, comments explaining math.

START WITH: 
import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')  # Force CPU execution for stability
jax.config.update('jax_enable_x64', False)

{end_statement}

Output 300-400 lines of valid Python code only.Output ONLY valid Python code with # comments.
- NO markdown code fences (``` or ```python)
- NO explanatory paragraphs or prose
- NO example usage sections
- NO numbered lists or bullet points
- Every line must be executable Python or a # comment
- Do NOT explain what the code does - just write the code Every line must be valid Python or a # comment."""

        # Stream the code generation
        print("\n" + "="*80)
        print(" BAYESIAN MODEL CODE (streaming):")
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
        
        logger.info(f"[OK] Generated {len(code.splitlines())} lines of model code")
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
            "adstock transformation geometric delayed exponential implementation",
            "saturation transformation hill logistic curve",
            "numpyro inference adam optimizer training loop",
        ]
        
        # Add MCMC-specific vectorization queries
        if backend == "numpyro_nuts":
            queries.extend([
                "jax vmap vectorized array operations parallel",
                "numpyro MCMC vectorized channels array processing",
                "jnp.stack expand broadcasting multiple channels",
                "fast MCMC parallel chains numpyro configuration"
            ])
        # Add advanced structure queries when TVP or correlated priors are used (default on)
        adv = getattr(spec, "advanced", None)
        use_tvp = getattr(adv, "time_varying_parameters", True) if adv else True
        use_correlated = getattr(adv, "correlated_channel_priors", True) if adv else True
        if use_tvp:
                queries.extend([
                    "numpyro time varying parameters random walk",
                    "time varying coefficient MCMC state space",
                    "jax lax scan random walk prior"
                ])
        if use_correlated:
                queries.extend([
                    "numpyro MultivariateNormal Cholesky LKJ",
                    "LKJCholesky correlation matrix prior",
                    "correlated prior Cholesky factor numpyro"
                ])
        
        examples = []
        if not self.rag:
            return ""
        
        for q in queries:
            try:
                # Fix: Use retrieve() instead of search() - RAG backend has retrieve() method
                results = self.rag.retrieve(q, n_results=3)
                if results:
                    # Extract output from dict format
                    for r in results:
                        if isinstance(r, dict):
                            examples.append(r.get('output', ''))
                        else:
                            examples.append(str(r))
            except Exception as e:
                logger.debug(f"RAG retrieval failed for '{q}': {e}")
                continue
        
        if not examples:
            return ""
        
        return f"""
� PRODUCTION MMM & API EXAMPLES - USE THESE EXACT PATTERNS:
{'��' * 80}
{chr(10).join([f" Example {i+1}:{chr(10)}{ex[:2000]}{chr(10)}{'' * 80}" for i, ex in enumerate(examples[:8])])}
{'��' * 80}
NOTE: Learn the patterns (vectorization, API usage, structure) but adapt them to your specific spec.
"""
    
    def _get_few_shot_examples(self) -> str:
        """Return 10 working few-shot examples for MCMC MMM"""
        return """
����������������������������������������������������������������������������������������������������������������������������������������������������������������
                MMM-SPECIFIC FEW-SHOT EXAMPLES (BAYESIAN MODELING)
����������������������������������������������������������������������������������������������������������������������������������������������������������������

These examples show MARKETING MIX MODEL patterns using NumPyro + JAX.

KEY MMM CONCEPTS IN THESE EXAMPLES:
  � ADSTOCK: Marketing impact persists (geometric decay)
  � SATURATION: Diminishing returns (Hill curve)
  � VECTORIZATION: Process all channels at once (fast!)
  � BAYESIAN PRIORS: Encode domain knowledge
  � MANIFEST OUTPUT: Save metadata for downstream modules

����������������������������������������������������������������������������������������������������������������������������������������������������������������

========== FEW-SHOT EXAMPLE 1: Basic scaling pattern (MMM preprocessing) ==========
# WHY: Marketing spend has huge range (TV: $1M, email: $100)
# Scale to [0, 1] so priors are meaningful across all channels
# ALWAYS scale inputs first!
channel_data = jnp.stack([data_dict[name] for name in channel_names])
channel_data = channel_data / jnp.max(channel_data, axis=1, keepdims=True)
target_scale = jnp.max(data_dict['outcome'])
target_scaled = data_dict['outcome'] / target_scale

========== FEW-SHOT EXAMPLE 2: Vectorized channel processing (MMM efficiency) ==========
# WHY: MMM models have 10-50+ channels. Loops are SLOW!
# Use jax.vmap to process ALL channels in parallel
def transform_channel(ch_data, decay, alpha, coef):
    x = geometric_adstock(ch_data, decay, 8)
    x = x / jnp.max(x)
    x = hill_saturation(x, alpha, 0.5)
    return coef * x
contributions = jax.vmap(transform_channel)(channel_data, decays, alphas, coefs)
mu += jnp.sum(contributions, axis=0)

========== FEW-SHOT EXAMPLE 3: Array parameter sampling (MMM vectorization) ==========
# WHY: Each channel has its own adstock decay, saturation curve, coefficient
# Sample ALL parameters as arrays (n_channels,) instead of loops
n_channels = len(channel_names)
decays = numpyro.sample('decays', dist.Beta(3, 3).expand([n_channels]))
alphas = numpyro.sample('alphas', dist.Beta(2, 2).expand([n_channels]))
coefs = numpyro.sample('coefs', dist.HalfNormal(0.1).expand([n_channels]))

========== FEW-SHOT EXAMPLE 4: Correct priors for scaled data (MMM regularization) ==========
# WHY: Scaled data [0, 1] �� outcomes and effects are small
# Use priors that match the scale: HalfNormal(0.1) instead of HalfNormal(1)
baseline = numpyro.sample('baseline', dist.Normal(0, 0.1))
sigma = numpyro.sample('sigma', dist.HalfNormal(0.1))

========== FEW-SHOT EXAMPLE 5: Geometric adstock with scan (MMM carry-over effect) ==========
# WHY: Ads today �� sales today + next week + week after (decay over time)
# geometric_adstock models this with decay parameter α (alpha)
def geometric_adstock(x: jnp.ndarray, alpha: float, max_lag: int = 8) -> jnp.ndarray:
    def scan_fn(carry, x_t):
        adstocked = alpha * carry + x_t
        return adstocked, adstocked
    _, result = jax.lax.scan(scan_fn, 0.0, x)
    return result

========== FEW-SHOT EXAMPLE 6: Hill saturation (MMM diminishing returns) ==========
# WHY: First $1000 of spend >> last $1000 (diminishing returns)
# Hill function models S-curve: flat at low spend, steep rise, then plateau
def hill_saturation(x: jnp.ndarray, alpha: float, beta: float = 0.5) -> jnp.ndarray:
    return (x ** alpha) / (x ** alpha + beta ** alpha)

========== FEW-SHOT EXAMPLE 7: MCMC setup with spec params (Bayesian inference) ==========
# WHY: Bayesian MMM quantifies uncertainty (95% credible intervals on ROI)
# NUTS sampler explores posterior distribution of all parameters
numpyro.set_host_device_count(num_chains)
nuts_kernel = NUTS(mmm_model, target_accept_prob=0.8, max_tree_depth=10)
mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=True)
mcmc.run(rng_key, data_dict=data_dict, n_obs=n_obs)

========== FEW-SHOT EXAMPLE 8: Predictions WITHOUT target leakage (Critical!) ==========
# WHY: MMM must predict outcome from channels ONLY (no cheating with actual outcome!)
# Remove target from data_dict before generating predictions
samples = mcmc.get_samples()
# CRITICAL FIX: Don't pass target to avoid data leakage!
# Save target_scale before removing target from data_dict
target_scale_value = target_scale  # Already computed during training
data_dict_no_target = {k: v for k, v in data_dict.items() if k != '{target_name}'}
# For prediction, pass target as None
data_dict_pred = data_dict_no_target.copy()
data_dict_pred['{target_name}'] = jnp.zeros(n_obs)  # Dummy target for scaling
predictive = Predictive(mmm_model, samples)
preds = predictive(rng_key, data_dict=data_dict_pred, n_obs=n_obs)
# Unscale for R2 calculation  
predicted = np.array(preds['obs'].mean(axis=0)) * target_scale_value

========== FEW-SHOT EXAMPLE 9: Controls vectorized (MMM baseline factors) ==========
# WHY: MMM must control for non-marketing factors (pricing, seasonality, holidays)
# Add controls with their own coefficients to isolate marketing effects
if all(c in data_dict for c in control_names):
    control_data = jnp.stack([data_dict[name] for name in control_names])
    control_data = control_data / jnp.max(control_data, axis=1, keepdims=True)
    control_coefs = numpyro.sample('control_coefs', dist.Normal(0, 0.1).expand([len(control_names)]))
    mu += jnp.sum(control_data * control_coefs[:, None], axis=0)

========== FEW-SHOT EXAMPLE 10: R² calculation (MMM model fit metric) ==========
# WHY: R² tells us % of outcome variance explained by marketing + controls
# R² > 0.8 = good fit, R² > 0.9 = excellent fit for MMM
# CRITICAL: BOTH actual AND predicted must be on SAME (original) scale. If you scaled
# data in run_modeling(), data_dict[outcome] is scaled—so set actual = data_dict[outcome] * target_scale
# right after the scale block and use that for R2 and CSV. Predictions: predicted = preds['obs'].mean(axis=0) * target_scale.
actual = np.array(data_dict['outcome']) * (target_scale if needs_scale else 1.0)  # original scale
predicted = np.array(preds['obs'].mean(axis=0)) * target_scale
ss_res = np.sum((actual - predicted)**2)
ss_tot = np.sum((actual - actual.mean())**2)
r2 = float(1 - ss_res / ss_tot)
"""
    
    def _get_advanced_model_instructions(self, spec: MMMSpec) -> str:
        """Return prompt block for TVPs and/or correlated (multivariate) priors. Default on for agentic modeling."""
        adv = getattr(spec, "advanced", None)
        # When advanced is omitted, default to full advanced (TVP + correlated priors)
        use_tvp = getattr(adv, "time_varying_parameters", True) if adv else True
        use_correlated = getattr(adv, "correlated_channel_priors", True) if adv else True
        if not use_tvp and not use_correlated:
            return ""
        parts = [
            """
CRITICAL - YOU MUST IMPLEMENT ADVANCED MODEL STRUCTURE:
Do NOT use static independent priors (e.g. coefs = dist.HalfNormal(0.1).expand([n_channels])).
The REFERENCE EXAMPLE below shows static coefs only for structure; you MUST replace channel coefficient sampling with the patterns below.
"""
        ]
        if use_correlated:
            parts.append("""
1) CORRELATED CHANNEL PRIORS (MANDATORY) - use LKJ Cholesky + MultivariateNormal pattern:
# Use concentration=3.0 or higher for init stability (avoids "Out-of-support" / "Cannot find valid initial parameters")
# Replace any "coefs = numpyro.sample('coefs', dist.HalfNormal(...).expand([n_channels]))" with:
L_channel = numpyro.sample('L_channel', dist.LKJCholesky(dimension=n_channels, concentration=3.0))
sigma_coef = numpyro.sample('sigma_coef', dist.HalfNormal(0.2).expand([n_channels]))
z_channel = numpyro.sample('z_channel', dist.Normal(0, 1).expand([n_channels]))
# Correlated coefficients (positive via softplus): coefs shape (n_channels,)
coefs = jnp.sqrt(sigma_coef**2) * (L_channel @ z_channel)
coefs = jnp.nn.softplus(coefs)  # ensure positive channel effects
# Then use coefs in transform_channel / contributions as usual (coefs[i] per channel).
""")
        if use_tvp:
            tvp_scale = (getattr(adv, "tvp_prior_scale", None) or "half_normal(0.05)") if adv else "half_normal(0.05)"
            parts.append(f"""
2) TIME-VARYING PARAMETERS (MANDATORY) - coefficients vary over time (random walk):
# Sample baseline level and innovation scale. Coefs evolve as random walk.
coef_0 = coefs  # from step 1 (correlated) or sample('coef_0', dist.HalfNormal(0.2).expand([n_channels]))
sigma_tvp = numpyro.sample('sigma_tvp', dist.HalfNormal(0.05).expand([n_channels]))
# Random walk: innovations (n_channels, n_obs-1), then cumsum to get coefs over time
innovations = numpyro.sample('coef_innovations', dist.Normal(0, 1).expand([n_channels, n_obs - 1]))
coefs_over_time = jnp.concatenate([coef_0[:, None], coef_0[:, None] + jnp.cumsum(sigma_tvp[:, None] * innovations, axis=1)], axis=1)  # (n_channels, n_obs)
# PRODUCTION RULE: vmap(transform_channel) expects 4th arg shape (n_channels,) only. Do NOT pass coefs_over_time into vmap.
# Two-step: (1) transformed_channel = jax.vmap(transform_channel)(channel_data, decays, alphas, jnp.ones(n_channels))  # shape (n_channels, n_obs)
#           (2) contributions = jnp.sum(coefs_over_time * transformed_channel, axis=0)  # (n_obs,)
""")
        parts.append("\n" + "=" * 80 + "\n")
        return "".join(parts)

    def _generate_mcmc_code_simple(self, spec: MMMSpec, spec_yaml: str, reasoning: dict = None) -> str:
        """Generate MCMC code using simple example-first approach with COT reasoning"""
        outcome = spec.outcome
        channels = [ch.name for ch in spec.channels]
        controls = [c.name if hasattr(c, 'name') else c for c in spec.controls] if spec.controls else []
        
        # Get adstock/saturation types from first channel's transform
        adstock_type = "geometric"
        saturation_type = "hill"
        if spec.channels and hasattr(spec.channels[0], 'transform'):
            if hasattr(spec.channels[0].transform, 'adstock'):
                adstock_type = spec.channels[0].transform.adstock.type
            if hasattr(spec.channels[0].transform, 'saturation'):
                saturation_type = spec.channels[0].transform.saturation.type
        
        num_chains = getattr(spec.inference, 'num_chains', 1)
        num_warmup = getattr(spec.inference, 'num_warmup', 500)
        num_samples = getattr(spec.inference, 'num_samples', 500)
        
        few_shot_examples = self._get_few_shot_examples()
        
        # Prepend reasoning context if available
        reasoning_context = ""
        if reasoning:
            reasoning_context = f"""
========== REASONING ANALYSIS (FOLLOW THIS!) ==========

TRANSFORMATIONS TO IMPLEMENT:
{reasoning.get('transformation_functions', [])}

SCALING STRATEGY (CRITICAL):
{reasoning.get('scaling_strategy', 'Scale data for numerical stability')}

VECTORIZATION APPROACH (USE THIS):
{reasoning.get('vectorization', 'Use array operations, not loops')}

MODEL STRUCTURE:
{reasoning.get('model_structure', 'Standard MMM structure')}

PRIORS TO USE:
{reasoning.get('priors', {})}

EXECUTION ORDER (FOLLOW EXACTLY):
"""
            for i, step in enumerate(reasoning.get('execution_order', []), 1):
                reasoning_context += f"\n{i}. {step}"
            
            reasoning_context += f"""

ISSUES TO AVOID:
{', '.join(reasoning.get('potential_issues', []))}

VALIDATION CHECKS:
{', '.join(reasoning.get('validation', []))}

=======================================================

"""
        
        advanced_block = self._get_advanced_model_instructions(spec)
        advanced_checklist = "\n6. **MUST use the ADVANCED patterns above for channel coefficients** (correlated LKJ Cholesky and/or time-varying random walk) - do NOT copy the static \"coefs = HalfNormal(0.1).expand\" from the reference example." if advanced_block.strip() else ""
        ref_note = "; for channel coefs you MUST use the ADVANCED code patterns above, not the static coefs here" if advanced_block.strip() else ""
        _pn = "param_name"  # so REQUIREMENTS can show literal param_name in f-string without NameError
        prompt = f"""{reasoning_context}Generate sophisticated NumPyro MCMC code by LEARNING FROM these patterns below.

IMPORTANT: These are PATTERNS to learn from, not templates to copy. Understand the concepts
(vectorization, scaling, transformations, Bayesian structure) and apply them intelligently
to create production-grade code for the spec.

{few_shot_examples}

SPEC TO IMPLEMENT:
- Outcome: {outcome}
- Channels: {channels} ({len(channels)} channels)
- Controls: {controls} ({len(controls) if controls else 0} controls)
- Adstock: {adstock_type}
- Saturation: {saturation_type}
- MCMC: {num_chains} chain(s), {num_warmup} warmup, {num_samples} samples
{advanced_block}
GENERATE SOPHISTICATED CODE that:
1. Uses vectorization (jax.vmap) for all {len(channels)} channels
2. Implements proper scaling and transformations
3. Includes comprehensive error handling
4. Saves complete manifests for downstream modules
5. Follows the patterns above but adapts them to this specific spec
{advanced_checklist}
REFERENCE EXAMPLE (structure only - adstock, saturation, baseline, controls{ref_note}):

```python
import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', False)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_feasible
import pandas as pd
import numpy as np

def geometric_adstock(x: jnp.ndarray, alpha: float, max_lag: int = 8) -> jnp.ndarray:
    '''Geometric adstock with JAX scan'''
    def scan_fn(carry, x_t):
        adstocked = alpha * carry + x_t
        return adstocked, adstocked
    _, result = jax.lax.scan(scan_fn, 0.0, x)
    return result

def hill_saturation(x: jnp.ndarray, alpha: float, beta: float = 0.5) -> jnp.ndarray:
    '''Hill saturation transformation'''
    return (x ** alpha) / (x ** alpha + beta ** alpha)

def mmm_model(data_dict: dict, n_obs: int):
    '''VECTORIZED MMM model - processes all channels in parallel'''
    # SPEC: Channels to model
    channel_names = ['tv_spend', 'radio_spend', 'digital_spend']
    n_channels = len(channel_names)
    
    # SCALE ALL INPUTS TO [0, 1] - CRITICAL FOR NUMERICAL STABILITY
    channel_data = jnp.stack([data_dict[name] for name in channel_names])
    channel_data = channel_data / jnp.max(channel_data, axis=1, keepdims=True)  # Scale each channel
    
    # Use .get() to make target optional (None during prediction)
    target_data = data_dict.get('revenue')
    target_scale = jnp.max(target_data) if target_data is not None else 1.0
    target_scaled = target_data / target_scale if target_data is not None else None
    
    # Baseline
    baseline = numpyro.sample('baseline', dist.Normal(0, 0.1))
    mu = jnp.ones(n_obs) * baseline
    
    # Sample ALL channel parameters as arrays (not loops!)
    # With scaled inputs, all parameters should be O(0.1)
    decays = numpyro.sample('decays', dist.Beta(3, 3).expand([n_channels]))
    alphas = numpyro.sample('alphas', dist.Beta(2, 2).expand([n_channels]))
    coefs = numpyro.sample('coefs', dist.HalfNormal(0.1).expand([n_channels]))
    
    # Define transformation function for ONE channel (already scaled to [0,1])
    def transform_channel(ch_data, decay, alpha, coef):
        x = geometric_adstock(ch_data, decay, 8)
        x = x / jnp.max(x)  # Renormalize after adstock
        x = hill_saturation(x, alpha, 0.5)
        return coef * x
    
    # Apply to ALL channels using vmap (parallel, vectorized)
    contributions = jax.vmap(transform_channel)(channel_data, decays, alphas, coefs)
    mu += jnp.sum(contributions, axis=0)
    
    # Controls (if any) - VECTORIZED and SCALED
    control_names = ['holiday', 'seasonality']
    if all(c in data_dict for c in control_names):
        control_data = jnp.stack([data_dict[name] for name in control_names])
        control_data = control_data / jnp.max(control_data, axis=1, keepdims=True)  # Scale each control
        control_coefs = numpyro.sample('control_coefs', dist.Normal(0, 0.1).expand([len(control_names)]))
        mu += jnp.sum(control_data * control_coefs[:, None], axis=0)
    
    # Likelihood - everything is now scaled to [0, 1]
    sigma = numpyro.sample('sigma', dist.HalfNormal(0.1))
    numpyro.sample('obs', dist.Normal(mu, sigma), obs=target_scaled)

def run_modeling(data_path: str):
    '''Entry point - loads data, runs MCMC, returns results dict'''
    print("Loading data...")
    df = pd.read_csv(data_path)
    data_dict = {{col: jnp.array(df[col].values) for col in df.columns}}
    n_obs = len(df)
    print(f"Loaded {{n_obs}} observations")
    # CHECK: Is data already scaled? If not, scale to [0,1] for MCMC stability (avoids init-params error)
    outcome_col = '{outcome}'
    target_scale = float(jnp.max(data_dict[outcome_col])) if outcome_col in data_dict else 1.0
    needs_scale = False
    for key in list(data_dict.keys()):
        col = data_dict[key]
        mx, mn = float(jnp.max(col)), float(jnp.min(col))
        if mx > 1.0 or mn < 0.0:
            needs_scale = True
            break
    if needs_scale:
        for key in list(data_dict.keys()):
            col = data_dict[key]
            mx = jnp.max(jnp.abs(col))
            if mx > 0:
                data_dict[key] = col / mx
        print("Scaled all columns to [0,1] for MCMC stability")
    else:
        print("Data already scaled to [0,1], skipping scale")
    # Actual on ORIGINAL scale for R2 (if we scaled, data_dict[outcome] is scaled—unscale it)
    actual = np.array(data_dict[outcome_col]) * (target_scale if needs_scale else 1.0)
    
    print("Setting up MCMC...")
    numpyro.set_host_device_count(4)
    nuts_kernel = NUTS(mmm_model, init_strategy=init_to_feasible(), target_accept_prob=0.8, max_tree_depth=10)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=500, num_chains=4, progress_bar=True)
    
    print("Running MCMC...", flush=True)
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key, data_dict=data_dict, n_obs=n_obs)
    mcmc.print_summary()
    
    print("Computing predictions...")
    samples = mcmc.get_samples()
    # CRITICAL: Don't pass target to prediction. actual already set above on original scale.
    data_dict_pred = {{k: v for k, v in data_dict.items() if k != outcome_col}}
    predictive = Predictive(mmm_model, samples, return_sites=['obs'])
    preds = predictive(rng_key, data_dict=data_dict_pred, n_obs=n_obs)
    
    print("Calculating R2...")
    # Unscale predictions to original scale for R2
    predicted = np.array(preds['obs'].mean(axis=0)) * target_scale
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - actual.mean())**2)
    r2 = float(1 - ss_res / ss_tot)
    print(f"R2 Score: {{r2:.4f}}")
    
    # Save results to CSV (JAX arrays don't pickle well!)
    # 1. Save predictions
    predictions_df = pd.DataFrame({{
        'actual': actual,
        'predicted': predicted
    }})
    predictions_df.to_csv('module_3_predictions.csv', index=False)
    
    # 2. Save parameters to CSV. Use this pattern so every column has length num_samples (axis 0):
{_PARAMS_SAVE_REF}
    # 3. Save metadata
    metadata_df = pd.DataFrame({{'metric': ['r2'], 'value': [r2]}})
    metadata_df.to_csv('module_3_metadata.csv', index=False)
    
    # 4. Save a JSON manifest for downstream modules
    import json
    manifest = {{
        'predictions_path': 'module_3_predictions.csv',
        'params_path': 'module_3_params.csv',
        'metadata_path': 'module_3_metadata.csv',
        'r2': r2,
        'n_params': len(samples),
        'n_samples': len(predicted)
    }}
    with open('module_3_results.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("Results saved to module_3_predictions.csv, module_3_params.csv, module_3_metadata.csv, module_3_results.json")
    
    return {{
        'params': {{k: np.array(v) for k, v in samples.items()}},
        'actual': actual,
        'predicted': predicted,
        'r2': r2,
        'predictions_path': 'module_3_predictions.csv',
        'params_path': 'module_3_params.csv',
        'metadata_path': 'module_3_metadata.csv',
        'manifest_path': 'module_3_results.json'
    }}
```

LEARN FROM THE EXAMPLE STRUCTURE:
1. Study the transformation functions (geometric_adstock, hill_saturation) - understand the math
2. Study the model structure (baseline, vectorized channels, controls, likelihood)
3. Study the MCMC setup and execution flow
4. Study the prediction and R2 calculation (unscaled!)
5. **CRITICAL PATTERNS**: 
   - `x = x / jnp.max(x)` after adstock (prevents numerical instability)
   - Scale target to [0,1] before likelihood (makes priors work correctly)
   - Use jax.vmap for vectorization (10-100x faster than loops)
   - Remove target from data_dict before prediction (no data leakage)
   - When building pd.DataFrame from MCMC samples or predictions: every column MUST be 1D. If an array is 2D or 3D+, flatten or slice it (e.g. np.ravel or one 1D column per index with np.ndindex) so pandas never receives 2D column values—otherwise ValueError("Per-column arrays must each be 1-dimensional").

NOW ADAPT TO YOUR SPEC:
1. channel_names = {channels}
2. outcome: data_dict['{outcome}'] (change 'revenue' to '{outcome}')
3. control_names = {controls} (if empty, remove control block entirely)
4. num_chains={num_chains}, num_warmup={num_warmup}, num_samples={num_samples}
5. Keep adstock as geometric_adstock (or add {adstock_type} if different)
6. Keep saturation as hill_saturation (or add {saturation_type} if different)

REQUIREMENTS (apply patterns from examples above):
- CHECK AND SCALE: In run_modeling(), immediately after building data_dict from the CSV, CHECK if data is already scaled (e.g. for each column, max <= 1 and min >= 0). If any column has max > 1 or min < 0, scale that column to [0,1] by dividing by max(abs(column)) (avoid div by zero). Save target_scale = max(outcome) BEFORE scaling for R2 unscaling later. Print "Data already scaled, skipping" or "Scaled N columns to [0,1]" so logs show what happened. This avoids "Cannot find valid initial parameters" when raw data is passed.
- Use jax.vmap for vectorized channel processing (NO loops!). If you pass in_axes to vmap, use in_axes=(0,0,0,0)—batch over axis 0 for all args (channel_data, decays, alphas, coefs).
- PRODUCTION INPUT: vmap(transform_channel) 4th argument must be (n_channels,) only. For time-varying coefs (n_channels, n_obs), do NOT pass them to vmap. Use: transformed = vmap(transform_channel)(channel_data, decays, alphas, jnp.ones(n_channels)); contributions = jnp.sum(coefs_over_time * transformed, axis=0).
- Sample parameters as arrays with .expand([n_channels])
- Use small priors (0.1 scale) for scaled data
- Use scan-based adstock and hill saturation
- MCMC setup with spec's num_chains/warmup/samples. Use init_strategy=init_to_feasible() in NUTS so NumPyro finds valid initial parameters; if L_channel still fails, use init_to_value(values={{"L_channel": jnp.eye(n_channels)}}) (identity Cholesky).
- R2 scale match: After the scale block, set actual = np.array(data_dict[outcome_col]) * (target_scale if needs_scale else 1.0) and use this for R2 and predictions CSV. Unscale predictions: predicted = preds['obs'].mean(axis=0) * target_scale. Both must be on original scale or R2 will be wildly wrong (huge negative).
- Vectorize controls with jnp.stack
- Correct R2 formula on unscaled data (actual and predicted same scale)
- Save complete manifest (predictions_path, params_path, metadata_path, channel_names, param_mapping)
- Output 150+ lines of production-grade code
- MUST return dict with params/actual/predicted/r2 and all file paths
- YOU must write the logic that saves MCMC samples to params CSV. Use this exact pattern (outer loop variable must be param_name so the inner f-string has it in scope). NumPyro samples have shape (num_samples, ...); every column must have length num_samples. Do NOT use shape[:-1] or param_values[idx] (causes "All arrays must be of the same length"). Correct pattern: params_data = {{}}; for param_name, param_values in samples.items(): (if param_values.ndim == 1: params_data[param_name] = param_values; else: for idx in np.ndindex(param_values.shape[1:]): params_data[f"{{_pn}}_{{'_'.join(map(str, idx))}}"] = param_values[(slice(None),) + idx]); params_df = pd.DataFrame(params_data).

OUTPUT ONLY PYTHON CODE - no markdown, no explanations."""

        print("  Generating code with LLM...")
        code = ""
        for chunk in self.llm.reason(prompt, stream=True):
            code += chunk
            print(chunk, end="", flush=True)
        print(f"\n\nGenerated {len(code.split(chr(10)))} lines")
        return self._clean_code(code)
    
    def _generate_svi_code_simple(self, spec: MMMSpec, spec_yaml: str) -> str:
        """Generate SVI code using simple example-first approach"""
        # TODO: Implement SVI simple method
        raise NotImplementedError("SVI backend not yet implemented with simple method. Use numpyro_nuts backend.")
    
    def _get_mcmc_training_guidance(self, spec: MMMSpec) -> str:
        """Generate MCMC-specific training guidance"""
        num_chains = getattr(spec.inference, 'num_chains', 1)
        num_warmup = getattr(spec.inference, 'num_warmup', 500)
        num_samples = getattr(spec.inference, 'num_samples', 500)
        outcome = spec.outcome
        
        return f"""## 3. MAIN ENTRY POINT: run_modeling function

YOU MUST COPY THIS STRUCTURE EXACTLY - Only change outcome variable name to '{outcome}':

```python
def run_modeling(data_path: str):
    '''Entry point: loads data, runs MCMC, returns results dict'''
    import pandas as pd
    import numpy as np
    from numpyro.infer import MCMC, NUTS, Predictive, init_to_feasible
    
    # 1. Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    data_dict = {{col: jnp.array(df[col].values) for col in df.columns}}
    n_obs = len(df)
    print(f"Loaded {{n_obs}} observations with {{len(data_dict)}} columns")
    
    # 2. CHECK AND SCALE: If any column has max > 1 or min < 0, scale to [0,1]; save target_scale and actual for R2
    outcome_col = '{outcome}'
    actual_original = np.array(data_dict[outcome_col]) if outcome_col in data_dict else np.zeros(n_obs)
    target_scale = float(jnp.max(data_dict[outcome_col])) if outcome_col in data_dict else 1.0
    needs_scale = any(float(jnp.max(data_dict[k])) > 1.0 or float(jnp.min(data_dict[k])) < 0.0 for k in data_dict)
    if needs_scale:
        for key in list(data_dict.keys()):
            col = data_dict[key]
            mx = jnp.max(jnp.abs(col))
            if mx > 0:
                data_dict[key] = col / mx
        print("Scaled all columns to [0,1] for MCMC stability")
    else:
        print("Data already scaled, skipping scale")
    
    # 3. MCMC configuration  
    print("Setting up MCMC...")
    numpyro.set_host_device_count({num_chains})
    nuts_kernel = NUTS(mmm_model, init_strategy=init_to_feasible(), target_accept_prob=0.8, max_tree_depth=10)
    mcmc = MCMC(nuts_kernel, num_warmup={num_warmup}, num_samples={num_samples}, num_chains={num_chains}, progress_bar=True)
    
    # 4. Run MCMC
    print("Running MCMC...", flush=True)
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key, data_dict=data_dict, n_obs=n_obs)
    mcmc.print_summary()
    
    # 5. Get samples
    print("Getting posterior samples...")
    samples = mcmc.get_samples()
    
    # 6. Predictions - CRITICAL: Don't pass target!
    print("Computing predictions...")
    target_scale = float(jnp.max(data_dict['{outcome}']))
    actual = np.array(data_dict['{outcome}'])
    # Remove target from data_dict (model uses .get() so returns None)
    data_dict_pred = {{k: v for k, v in data_dict.items() if k != '{outcome}'}}
    predictive = Predictive(mmm_model, samples, return_sites=['obs'])
    preds = predictive(rng_key, data_dict=data_dict, n_obs=n_obs)
    
    # 7. Metrics (use actual_original and unscale predicted by target_scale for R2)
    print("Calculating metrics...")
    actual = actual_original
    predicted = np.array(preds['obs'].mean(axis=0)) * target_scale
    r2 = float(1 - np.sum((actual - predicted)**2) / np.sum((actual - actual.mean())**2))
    print(f"R2 Score: {{r2:.4f}}")
    
    # 8. Save results and return dict
    results = {{
        'params': {{k: np.array(v) for k, v in samples.items()}},
        'actual': actual,
        'predicted': predicted,
        'r2': r2
    }}
```

ABSOLUTE REQUIREMENTS:
1. Signature: def run_modeling(data_path: str) - NO OTHER PARAMETERS!
2. Must load data from data_path using pandas
3. Must call mmm_model function that you defined earlier
4. Must save results to CSV files (JAX arrays don't pickle well):
   - module_3_predictions.csv (actual, predicted)
   - module_3_params.csv (all MCMC parameters)
   - module_3_metadata.csv (r2 and other metrics)
5. Must return dict with params, actual, predicted, r2, and CSV paths
6. NO wrapper functions, NO config parameters
7. Use outcome variable '{outcome}' not hardcoded 'target_visits'"""
    
    def _get_svi_training_guidance(self, spec: MMMSpec) -> str:
        """Generate SVI-specific training guidance"""
        return """## 3. SVI TRAINING FUNCTION (60-80 lines)

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
   - Return samples, losses, svi, svi_state"""
    
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

