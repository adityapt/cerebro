"""
ExecutionValidator: Self-healing execution validation with minimal config injection

Key features:
- Runs code with minimal MCMC config for fast validation (10 samples vs 500)
- Auto-fixes errors using LLM + RAG + few-shot examples
- Automatically installs missing packages (ModuleNotFoundError)
- Retries up to max_retries times
- Uses REGEX-ONLY for simple fixes (return types, signatures)
"""
import logging
import traceback
import re
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import package manager for auto-installation
try:
    from cerebro.utils.package_manager import PackageManager
except ImportError:
    PackageManager = None


# Few-shot examples for common errors
FEW_SHOT_FIXES = {
    "AttributeError.*get_params": """
FIX: AutoNormal guide parameter extraction
WRONG:
params = guide.get_params(svi_state)

RIGHT:
raw_params = svi_result.params
params = guide.median(raw_params)
""",
    
    "parallel chains|set_host_device_count": """
FIX: MCMC parallel chains on CPU
Add at the beginning:
import numpyro
numpyro.set_host_device_count(1)  # For CPU parallel chains
""",
    
    "NaN|inf|numerical": """
FIX: Numerical stability - scale inputs to [0, 1]
WRONG:
channel_data = jnp.array(df[channel].values)

RIGHT:
channel_data = jnp.array(df[channel].values)
channel_data = channel_data / jnp.max(channel_data)  # Scale to [0, 1]
""",
    
    "All arrays must be of the same length": """
FIX: pandas DataFrame "All arrays must be of the same length" when saving MCMC params.
Cause: NumPyro samples have shape (num_samples, ...). Every column must have length num_samples (axis 0).
WRONG (columns end up with different lengths):
  for idx in np.ndindex(param_values.shape[:-1]):
      params_data[...] = param_values[idx]
RIGHT:
  for idx in np.ndindex(param_values.shape[1:]):
      params_data[f"{param_name}_{'_'.join(map(str, idx))}"] = param_values[(slice(None),) + idx]
So we slice along axis 0 (samples) and iterate over the rest; each column gets length num_samples.
""",
    "vmap.*inconsistent sizes|inconsistent sizes.*vmap": """
FIX: vmap "inconsistent sizes" - production rule: 4th arg to vmap(transform_channel) must be shape (n_channels,) only.
WRONG (coefs_over_time is (n_channels, n_obs)):
  contributions = jax.vmap(transform_channel)(channel_data, decays, alphas, coefs_over_time)
RIGHT for time-varying coefs:
  transformed_channel = jax.vmap(transform_channel)(channel_data, decays, alphas, jnp.ones(n_channels))
  contributions = jnp.sum(coefs_over_time * transformed_channel, axis=0)
For static coefs keep: vmap(transform_channel)(channel_data, decays, alphas, coefs) with coefs shape (n_channels,).
""",
    "initial parameters|Cannot find valid initial": """
FIX: NumPyro "Cannot find valid initial parameters"
Cause: Priors too tight or init values outside support. Do ONE or more:
1. Use init_to_median() or init_to_feasible() in MCMC:
   kernel = NUTS(model, init_strategy=init_to_median(num_samples=10))
   or: init_strategy=init_to_feasible()
2. If using LKJCholesky for L_channel: use concentration=8.0 or higher (not 1 or 2) to avoid "Out-of-support".
3. Custom init values for stubborn sites (e.g. L_channel): use init_to_value so MCMC starts at a valid point:
   from numpyro.infer import init_to_value
   init_vals = init_to_value(values={"L_channel": jnp.eye(n_channels)})  # identity Cholesky = valid
   kernel = NUTS(model, init_strategy=init_vals, ...)
   Ensure n_channels is in scope where NUTS is built.
4. Widen priors so init is valid (e.g. HalfNormal(0.5) instead of 0.1 for scale)
5. Ensure positive bounds: sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))
6. Scale all inputs to [0,1] before the model so likelihood isn't extreme
Return the COMPLETE fixed run_modeling code with the init_strategy and/or prior changes applied.
""",
}


class ExecutionValidator:
    """Execution validation with self-healing and minimal config injection"""
    
    def __init__(self, llm, max_retries: int = 15, rag=None, auto_install_packages: bool = True):
        self.llm = llm
        self.max_retries = max_retries
        self.rag = rag
        self.auto_install_packages = auto_install_packages
        # Initialize package manager for auto-installation
        if PackageManager and auto_install_packages:
            self.package_manager = PackageManager(auto_install=True, ask_user=False)
        else:
            self.package_manager = None
    
    def validate_and_fix(
        self,
        code: str,
        spec,
        module_name: str,
        entry_point: str,
        test_args: tuple
    ) -> Tuple[str, Dict]:
        """
        Execute code and auto-fix errors
        
        Returns:
            Tuple of (fixed_code, fixes_applied)
        """
        fixes_applied = {}
        
        # CRITICAL: Fix return type BEFORE execution using regex-only
        code = self._ensure_return_type_annotation(code, entry_point)
        
        # CRITICAL: Fix common syntax errors deterministically BEFORE asking LLM
        code, syntax_fixes = self._fix_common_syntax_errors(code)
        if syntax_fixes:
            fixes_applied['syntax_fixes'] = syntax_fixes
        
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Validation attempt {attempt}/{self.max_retries} for {module_name}")
            
            try:
                # Ensure code is not None
                if not code or not isinstance(code, str):
                    raise ValueError(f"Invalid code type: {type(code)}")
                
                # PRE-EXECUTION: Scan for missing packages and install them
                if self.package_manager and attempt == 1:
                    missing_packages = self._scan_and_install_packages(code)
                    if missing_packages:
                        logger.info(f"  [PRE-INSTALL] Installed {len(missing_packages)} packages: {', '.join(missing_packages)}")
                        fixes_applied['pre_installed_packages'] = missing_packages
                
                exec_globals = {}
                
                # Inject minimal config for MCMC modules (validation only; full run uses spec config)
                if module_name == 'modeling' and 'MCMC' in code:
                    logger.info("Injecting minimal MCMC config for fast validation (10 warmup, 10 samples). Full run will use spec config.")
                    code_to_exec = self._inject_minimal_mcmc_config(code)
                else:
                    code_to_exec = code
                
                # So MCMC progress shows when stdout is piped (e.g. tee)
                if module_name == 'modeling':
                    import sys
                    try:
                        if hasattr(sys.stdout, 'reconfigure'):
                            sys.stdout.reconfigure(line_buffering=True)
                        if hasattr(sys.stderr, 'reconfigure'):
                            sys.stderr.reconfigure(line_buffering=True)
                        sys.stdout.flush()
                        sys.stderr.flush()
                    except Exception:
                        pass
                
                # Execute
                exec(code_to_exec, exec_globals)
                
                # Check entry point exists
                if entry_point not in exec_globals:
                    raise ValueError(f"Entry point '{entry_point}' not found")
                
                # Try calling it
                result = exec_globals[entry_point](*test_args)
                
                # Success!
                logger.info(f" Execution validation passed for {module_name}")
                return code, fixes_applied
                
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                error_trace = traceback.format_exc()
                
                logger.warning(f"Attempt {attempt} failed: {error_type}: {error_msg}")
                
                # AUTO-INSTALL MISSING PACKAGES (ModuleNotFoundError)
                if error_type == 'ModuleNotFoundError' and self.package_manager:
                    package_name = self._extract_package_name(error_msg)
                    if package_name:
                        logger.info(f"  [AUTO-INSTALL] Detected missing package: {package_name}")
                        if self._install_missing_package(package_name):
                            logger.info(f"  [AUTO-INSTALL] Successfully installed {package_name}, retrying...")
                            fixes_applied[f'attempt_{attempt}'] = f'AUTO-INSTALLED: {package_name}'
                            # Don't modify code, just retry execution
                            continue  # Retry immediately without code changes
                        else:
                            logger.warning(f"  [AUTO-INSTALL] Failed to install {package_name}, will try LLM fix")
                
                if attempt < self.max_retries:
                    # Deterministic fix for vmap "inconsistent sizes" (modeling): wrong in_axes batch axis
                    if module_name == "modeling" and "vmap" in error_msg and "inconsistent sizes" in error_msg:
                        fixed = self._fix_vmap_in_axes(code)
                        if fixed != code:
                            code = fixed
                            logger.info("  [AUTO-FIX] Fixed vmap in_axes (batch axis 0) for inconsistent sizes; retrying.")
                            fixes_applied[f'attempt_{attempt}'] = "FIXED: vmap in_axes"
                            continue
                    # Deterministic fix for params DataFrame "All arrays must be of the same length" (axis 0 = num_samples)
                    if module_name == "modeling" and "All arrays must be of the same length" in error_msg:
                        fixed = self._fix_params_dataframe_same_length(code)
                        if fixed != code:
                            code = fixed
                            logger.info("  [AUTO-FIX] Fixed params DataFrame (shape[1:] + slice so columns length num_samples); retrying.")
                            fixes_applied[f'attempt_{attempt}'] = "FIXED: params_data same length"
                            continue
                    # Deterministic fix for NumPyro "Cannot find valid initial parameters" (modeling only)
                    if module_name == "modeling" and "initial parameters" in error_msg.lower():
                        # 1) Try adding/upgrading init_strategy in NUTS
                        use_feasible = "init_strategy" in code or "init_to_median" not in code
                        injected = self._inject_init_strategy(code, use_feasible=use_feasible)
                        if injected != code:
                            code = injected
                            logger.info("  [AUTO-FIX] Injected init strategy into NUTS for initial-parameters error; retrying.")
                            fixes_applied[f'attempt_{attempt}'] = "INJECTED: init_to_feasible" if use_feasible else "INJECTED: init_to_median"
                            continue
                        # 2) Init already present (or NUTS pattern not found); try softening LKJ prior (fixes L_channel out-of-support)
                        lkj_fixed = self._fix_lkj_concentration(code)
                        if lkj_fixed != code:
                            code = lkj_fixed
                            logger.info("  [AUTO-FIX] Softened LKJ concentration for initial-parameters error; retrying.")
                            fixes_applied[f'attempt_{attempt}'] = "SOFTENED: LKJ concentration"
                            continue
                        # 3) Try explicit init value for L_channel (identity Cholesky = valid starting point)
                        init_val_fixed = self._inject_init_to_value_lchannel(code)
                        if init_val_fixed != code:
                            code = init_val_fixed
                            logger.info("  [AUTO-FIX] Injected init_to_value(L_channel=jnp.eye) for initial-parameters error; retrying.")
                            fixes_applied[f'attempt_{attempt}'] = "INJECTED: init_to_value L_channel"
                            continue
                        logger.warning("  [AUTO-FIX] Init/LKJ/init_to_value had no match. Will try LLM fix.")
                    # Auto-fix using LLM + RAG + few-shot (for other errors or if injection not applied)
                    code = self._fix_error(
                        code, error_type, error_msg, error_trace, 
                        spec, module_name, entry_point
                    )
                    fixes_applied[f'attempt_{attempt}'] = f'{error_type}: {error_msg[:100]}'
                else:
                    logger.error(f"Max retries exceeded for {module_name}")
                    return code, fixes_applied
        
        return code, fixes_applied
    
    def _ensure_return_type_annotation(self, code: str, entry_point: str) -> str:
        """
        REGEX-ONLY fix for missing return type annotations
        Does NOT involve LLM to avoid code destruction
        """
        expected_returns = {
            'run_preprocessing': 'str',
            'run_exploration': 'dict',
            'run_modeling': 'dict',
            'run_diagnostics': 'dict',
            'run_optimization': 'dict',
            'run_visualization': 'dict'
        }
        
        if entry_point not in expected_returns:
            return code
        
        expected = expected_returns[entry_point]
        
        # Check if return type exists
        pattern = rf'def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*(->\s*\w+)?:'
        match = re.search(pattern, code)
        
        if not match:
            return code
        
        if match.group(1):  # Has return type
            # Check if it's correct
            current_type = match.group(1).strip('->: ')
            if current_type == expected:
                return code  # Already correct
            
            # Fix incorrect return type (regex only)
            logger.info(f"[REGEX] Fixing return type: {current_type} -> {expected}")
            code = re.sub(
                rf'(def\s+{re.escape(entry_point)}\s*\([^)]*\))\s*->\s*\w+\s*:',
                rf'\1 -> {expected}:',
                code
            )
        else:
            # Add missing return type (regex only)
            logger.info(f"[REGEX] Adding return type annotation: -> {expected}")
            code = re.sub(
                rf'(def\s+{re.escape(entry_point)}\s*\([^)]*\)):',
                rf'\1 -> {expected}:',
                code
            )
        
        return code
    
    def _inject_minimal_mcmc_config(self, code: str) -> str:
        """Inject minimal config for fast validation"""
        # Replace MCMC parameters with minimal values
        code_minimal = re.sub(r'num_warmup\s*=\s*\d+', 'num_warmup=10', code)
        code_minimal = re.sub(r'num_samples\s*=\s*\d+', 'num_samples=10', code_minimal)
        code_minimal = re.sub(r'num_chains\s*=\s*\d+', 'num_chains=1', code_minimal)
        
        # Add set_host_device_count if not present
        if 'set_host_device_count' not in code_minimal:
            # Add after imports
            import_end = code_minimal.find('\n\n')
            if import_end != -1:
                code_minimal = (
                    code_minimal[:import_end] +
                    '\nimport numpyro\nnumpyro.set_host_device_count(1)' +
                    code_minimal[import_end:]
                )
        
        logger.info("[INJECTED] Minimal MCMC config: 10 warmup, 10 samples, 1 chain")
        return code_minimal
    
    def _inject_init_strategy(self, code: str, use_feasible: bool = False) -> str:
        """Inject init_to_median or init_to_feasible into NUTS so NumPyro finds valid initial parameters."""
        if not code:
            return code
        # If we already have init_strategy, optionally upgrade to init_to_feasible (more robust)
        if use_feasible and "init_to_median" in code and "init_to_feasible" not in code:
            # Replace any init_strategy=init_to_median(...) with init_to_feasible()
            code = re.sub(
                r'init_strategy\s*=\s*init_to_median\s*\([^)]*\)',
                'init_strategy=init_to_feasible()',
                code,
                count=1
            )
            if "init_to_feasible" in code:
                # Ensure init_to_feasible is in the numpyro.infer import (check import line, not call site)
                match = re.search(r'from numpyro\.infer import [^\n]+', code)
                if match and 'init_to_feasible' not in match.group(0):
                    code = re.sub(
                        r'(from numpyro\.infer import [^\n]+)',
                        r'\1, init_to_feasible',
                        code,
                        count=1
                    )
                logger.info("[INJECTED] Replaced init_to_median with init_to_feasible() in NUTS")
                return code
        if "init_strategy" in code and not use_feasible:
            return code
        if "init_to_median" in code and not use_feasible:
            return code
        # Add import: append to first single-line numpyro.infer import (avoid breaking "import (\n...")
        if "init_to_feasible" not in code and "init_to_median" not in code:
            if "from numpyro.infer import" in code and "import (" not in code.split("from numpyro.infer import", 1)[1].split("\n")[0]:
                code = re.sub(
                    r'(from numpyro\.infer import [^\n()]+)(\n)',
                    r'\1, init_to_median, init_to_feasible\2',
                    code,
                    count=1
                )
            elif "from numpyro.infer import" in code:
                # Multi-line import: add new line
                code = re.sub(
                    r'(from numpyro\.infer import [^\n]+)',
                    r'\1\nfrom numpyro.infer import init_to_median, init_to_feasible',
                    code,
                    count=1
                )
            else:
                # No numpyro.infer import; add after "import numpyro" or at start of file
                if "import numpyro" in code:
                    code = re.sub(
                        r'(\bimport numpyro[^\n]*)',
                        r'\1\nfrom numpyro.infer import init_to_median, init_to_feasible',
                        code,
                        count=1
                    )
                else:
                    # Prepend after first line (shebang/encoding) or at start
                    first_newline = code.find('\n')
                    insert = first_newline + 1 if first_newline != -1 else 0
                    code = code[:insert] + "from numpyro.infer import init_to_median, init_to_feasible\n" + code[insert:]
        # NUTS(model, ...) or NUTS(model) -> add init_strategy (match with optional newlines)
        init_fn = "init_to_feasible()" if use_feasible else "init_to_median(num_samples=10)"
        if "init_strategy" not in code:
            # Match NUTS( <name> , or NUTS( <name> ) with any whitespace including newlines
            code = re.sub(
                r'NUTS\s*\(\s*(\w+)\s*,\s*',
                rf'NUTS(\1, init_strategy={init_fn}, ',
                code,
                count=1,
                flags=re.DOTALL
            )
        if "init_strategy" not in code:
            code = re.sub(
                r'NUTS\s*\(\s*(\w+)\s*\)',
                rf'NUTS(\1, init_strategy={init_fn})',
                code,
                count=1,
                flags=re.DOTALL
            )
        if "init_strategy" in code:
            logger.info("[INJECTED] %s into NUTS", init_fn)
        return code
    
    def _fix_vmap_in_axes(self, code: str) -> str:
        """
        Fix vmap 'inconsistent sizes' error: batch axis must be 0 for all args.
        When coef/channel_data have shape (n_channels, n_obs), in_axes must be (0,0,0,0)
        not (0,0,0,1). Replace trailing , 1) or ,1) in in_axes=(...) with , 0).
        """
        if not code or "in_axes" not in code:
            return code
        # Replace last axis value 1 with 0 in in_axes=(...); handles (0,0,0,1) and (0, 0, 0, 1)
        new_code = re.sub(
            r'(in_axes\s*=\s*\([^)]*),\s*1\s*\)',
            r'\1, 0)',
            code,
            count=1
        )
        if new_code != code:
            logger.info("[INJECTED] vmap in_axes: last axis 1 -> 0 (batch over axis 0)")
        return new_code
    
    def _fix_lkj_concentration(self, code: str) -> str:
        """Soften LKJ Cholesky prior: concentration=1/2/3 -> 8.0 for init stability (L_channel out-of-support)."""
        if not code or "LKJCholesky" not in code:
            return code
        new_code = re.sub(
            r'concentration\s*=\s*(?:1|2|3)(?:\.0)?\b',
            'concentration=8.0',
            code
        )
        if new_code != code:
            logger.info("[INJECTED] LKJCholesky concentration -> 8.0 for init stability")
        return new_code
    
    def _inject_init_to_value_lchannel(self, code: str) -> str:
        """When init_to_feasible and LKJ soften aren't enough: set explicit init value for L_channel (identity Cholesky)."""
        if not code or "L_channel" not in code or "init_to_value" in code:
            return code
        # Only replace init_to_feasible() with init_to_value(values={"L_channel": jnp.eye(n_channels)})
        if "init_strategy=init_to_feasible()" not in code and "init_strategy=init_to_feasible ()" not in code:
            return code
        new_code = re.sub(
            r'init_strategy\s*=\s*init_to_feasible\s*\(\s*\)',
            'init_strategy=init_to_value(values={"L_channel": jnp.eye(n_channels)})',
            code,
            count=1
        )
        if new_code == code:
            return code
        # Add init_to_value to numpyro.infer import if missing
        match = re.search(r'from numpyro\.infer import [^\n]+', new_code)
        if match and 'init_to_value' not in match.group(0):
            new_code = re.sub(
                r'(from numpyro\.infer import [^\n]+)',
                r'\1, init_to_value',
                new_code,
                count=1
            )
        if new_code != code:
            logger.info("[INJECTED] init_to_value(L_channel=jnp.eye(n_channels)) for valid init")
        return new_code
    
    def _fix_params_dataframe_same_length(self, code: str) -> str:
        """Fix params_data loop so every column has length num_samples (axis 0). Replace shape[:-1] with shape[1:] and param_values[idx] with param_values[(slice(None),) + idx]."""
        if not code or "params_data" not in code or "ndindex" not in code:
            return code
        new_code = code
        # Replace shape[:-1] with shape[1:] so we iterate over non-sample axes
        new_code = re.sub(
            r'param_values\.shape\[:\-1\]',
            'param_values.shape[1:]',
            new_code,
            count=1
        )
        # Replace param_values[idx] with param_values[(slice(None),) + idx] so each column has length num_samples
        new_code = re.sub(
            r'param_values\[idx\]',
            'param_values[(slice(None),) + idx]',
            new_code,
            count=1
        )
        if new_code != code:
            logger.info("[INJECTED] params_data: shape[1:] and (slice(None),)+idx for same-length columns")
        return new_code
    
    def _fix_error(
        self,
        code: str,
        error_type: str,
        error_msg: str,
        error_trace: str,
        spec,
        module_name: str,
        entry_point: str
    ) -> str:
        """Fix error using LLM + RAG + few-shot examples"""
        
        # Get relevant few-shot examples
        relevant_examples = []
        error_pattern = f"{error_type} {error_msg}".lower()
        for pattern, example in FEW_SHOT_FIXES.items():
            if re.search(pattern, error_pattern, re.IGNORECASE):
                relevant_examples.append(example)
        
        few_shot_section = "\n\n".join(relevant_examples) if relevant_examples else ""
        
        # Query RAG
        rag_examples = ""
        if self.rag:
            try:
                queries = [
                    f"{error_type} {error_msg[:50]} fix",
                    f"NumPyro MCMC {error_type} solution"
                ]
                results = []
                for q in queries:
                    rag_results = self.rag.retrieve(q, n_results=2)
                    if rag_results:
                        results.extend([r.get('output', '') for r in rag_results if r.get('output')])
                
                if results:
                    rag_examples = "\n\nRAG EXAMPLES:\n" + "\n---\n".join(results[:2])
            except Exception as e:
                logger.warning(f"RAG query failed: {e}")
        
        # Build fix prompt with signature requirement
        signature_requirement = self._get_signature_requirement(entry_point)
        
        # For modeling, send more code so NUTS/MCMC and priors are visible (init params fix needs them)
        snippet_len = 8000 if module_name == "modeling" else 2000
        code_snippet = ""
        if code and isinstance(code, str):
            code_snippet = code[:snippet_len]
            if len(code) > snippet_len:
                code_snippet += f"\n# ... ({len(code) - snippet_len} more chars truncated)"
        else:
            code_snippet = "# Code unavailable"
        
        # Error-specific hint for NumPyro initial-parameters failure
        initial_params_hint = ""
        if "initial parameters" in error_msg.lower() or "cannot find valid initial" in error_msg.lower():
            initial_params_hint = """
THIS ERROR MEANS: NumPyro could not find valid starting values. You MUST fix it by:
- Adding init_strategy to NUTS, e.g. kernel = NUTS(model, init_strategy=init_to_median(num_samples=10))
  (import: from numpyro.infer import init_to_median, init_to_feasible)
- And/or widening priors (e.g. HalfNormal(0.5) instead of 0.1) so the default init lies in the support
- And/or ensuring all data passed to the model is scaled (e.g. [0,1]) so likelihood is not NaN/inf
Return the COMPLETE fixed Python code.

"""
        
        prompt = f"""Fix the {module_name} code that has a runtime error.

ERROR:
{error_type}: {error_msg}

TRACEBACK (last 500 chars):
{error_trace[-500:]}
{initial_params_hint}
CURRENT CODE (first {snippet_len} chars):
```python
{code_snippet}
```

{signature_requirement}

{few_shot_section}

{rag_examples}

CRITICAL RULES:
1. Fix the error but preserve all working code
2. Do NOT regenerate from scratch - only fix the error
3. Use regex/numpy where possible instead of Python loops
4. Maintain the correct function signature
5. Return the COMPLETE fixed code with all imports

Output ONLY the COMPLETE fixed Python code. NO explanations, NO markdown fences."""
        
        # Get fix from LLM (non-streaming for simpler handling)
        try:
            fixed_code = self.llm.reason(prompt, stream=False)
            
            # Handle None or empty response
            if not fixed_code or not isinstance(fixed_code, str):
                logger.error(f"LLM returned invalid response type: {type(fixed_code)}")
                fixed_code = None
            
            cleaned_code = self._clean_code(fixed_code) if fixed_code else None
            
            # If LLM returned empty/None, keep original code
            if not cleaned_code or not cleaned_code.strip():
                logger.warning("LLM returned empty code, keeping original")
                # But if original is also None/empty, we're stuck - return a minimal stub
                if not code or not isinstance(code, str) or not code.strip():
                    logger.error("Both original and fixed code are empty! Returning minimal stub")
                    return self._generate_minimal_stub(entry_point)
                return code
            
            return cleaned_code
        except Exception as e:
            logger.error(f"Error getting LLM fix: {e}, keeping original code")
            # If original is also bad, return stub
            if not code or not isinstance(code, str):
                return self._generate_minimal_stub(entry_point)
            return code
        
    def _generate_minimal_stub(self, entry_point: str) -> str:
        """Generate a minimal working stub when all else fails"""
        stubs = {
            'run_preprocessing': """
import pandas as pd

def run_preprocessing(data_path: str) -> str:
    df = pd.read_csv(data_path)
    output_path = data_path.replace('.csv', '_preprocessed.csv')
    df.to_csv(output_path, index=False)
    return output_path
""",
            'run_exploration': """
import pandas as pd

def run_exploration(data_path: str) -> dict:
    df = pd.read_csv(data_path)
    return {'status': 'completed', 'rows': len(df)}
""",
            'run_modeling': """
import pandas as pd

def run_modeling(data_path: str) -> dict:
    return {'status': 'placeholder', 'params': {}}
""",
        }
        return stubs.get(entry_point, f"def {entry_point}(*args): return {{}}")
    
    def _fix_common_syntax_errors(self, code: str) -> Tuple[str, list]:
        """
        Deterministically fix common syntax errors using regex.
        This runs BEFORE asking LLM, so it's fast and reliable.
        
        Returns:
            Tuple of (fixed_code, list_of_fixes_applied)
        """
        import re
        fixes = []
        
        # Fix 1: Unmatched braces - remove standalone } or {
        # Common in f-strings when LLM confuses {{ }} with single braces
        original = code
        # Replace {{something}} in f-strings with {something}
        code = re.sub(r'\{\{([^}]+)\}\}', r'{\1}', code)
        if code != original:
            fixes.append("Fixed double braces {{}} to single {}")
            original = code
        
        # Fix 2: DISABLED - was removing valid closing braces in multi-line dicts
        # Don't blindly remove standalone { or } - they might be valid dict/set syntax
        # Let Python's ast.parse validate instead
        
        # Fix 3: Trailing commas in function definitions
        code = re.sub(r',(\s*\)):', r'\1:', code)
        if code != original:
            fixes.append("Removed trailing commas in function definitions")
        
        # Fix 4: Missing colons in function/class definitions
        code = re.sub(r'(def\s+\w+\([^)]*\)\s*(?:->\s*\w+)?)\s*\n', r'\1:\n', code)
        
        return code, fixes
    
    def _get_signature_requirement(self, entry_point: str) -> str:
        """Get the required function signature"""
        signatures = {
            'run_preprocessing': 'def run_preprocessing(data_path: str) -> str:',
            'run_exploration': 'def run_exploration(data_path: str) -> dict:',
            'run_modeling': 'def run_modeling(data_path: str) -> dict:',
            'run_diagnostics': 'def run_diagnostics(results_path: str) -> dict:',
            'run_optimization': 'def run_optimization(results_path: str) -> dict:',
            'run_visualization': 'def run_visualization(results_path: str) -> dict:',
        }
        
        if entry_point in signatures:
            return f"""
MANDATORY FUNCTION SIGNATURE:
{signatures[entry_point]}

DO NOT CHANGE THIS SIGNATURE.
"""
        return ""
    
    def _clean_code(self, code: str) -> str:
        """Remove markdown artifacts and extract Python code block if LLM added prose."""
        if not code or not isinstance(code, str):
            return ""
        code = code.strip()
        # If response looks like prose + code block, extract the largest ```python ... ``` block
        if "```python" in code and ("here is" in code.lower()[:200] or code.count("```") >= 2):
            blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", code, re.DOTALL)
            if blocks:
                # Prefer the longest block (likely the full fix)
                code = max(blocks, key=len).strip()
        # Remove outer markdown fences
        if code.startswith('```python'):
            code = code[len('```python'):].strip()
        elif code.startswith('```'):
            code = code[3:].strip()
        if code.endswith('```'):
            code = code[:-3].strip()
        return code
    
    def _extract_package_name(self, error_msg: str) -> Optional[str]:
        """
        Extract package name from ModuleNotFoundError message.
        
        Examples:
        - "No module named 'matplotlib'" -> "matplotlib"
        - "No module named 'jax'" -> "jax"
        - "No module named 'sklearn'" -> "scikit-learn" (map common aliases)
        """
        # Pattern: "No module named 'package_name'"
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_msg)
        if not match:
            return None
        
        package_name = match.group(1)
        
        # Map common import aliases to pip package names
        package_mapping = {
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'yaml': 'PyYAML',
        }
        
        return package_mapping.get(package_name, package_name)
    
    def _install_missing_package(self, package_name: str) -> bool:
        """
        Install a missing package using PackageManager.
        
        Returns:
            True if installation succeeded, False otherwise
        """
        if not self.package_manager:
            return False
        
        try:
            # Use silent=True to skip user confirmation for auto-install
            success = self.package_manager.install_package(package_name, silent=True)
            return success
        except Exception as e:
            logger.error(f"  ✗ Failed to install {package_name}: {e}")
            return False
    
    def _scan_and_install_packages(self, code: str) -> list:
        """
        Scan code for import statements and install missing packages proactively.
        
        Uses LLM to intelligently determine:
        1. Which imports are standard library (don't install)
        2. What the pip package name is for each import (handles aliases like sklearn -> scikit-learn)
        
        This catches packages that might be imported inside try-except blocks,
        which wouldn't raise exceptions to the ExecutionValidator.
        
        Returns:
            List of package names that were installed
        """
        if not self.package_manager:
            return []
        
        installed_packages = []
        
        # Find all import statements (handle both direct imports and "as" aliases)
        import_patterns = [
            r'import\s+(\w+)(?:\s+as\s+\w+)?',  # import package [as alias]
            r'from\s+(\w+)\s+import',  # from package import
        ]
        
        found_packages = set()
        for pattern in import_patterns:
            matches = re.findall(pattern, code)
            found_packages.update(matches)
        
        # Also handle "import arviz as az" - extract the actual package name
        arviz_pattern = r'import\s+(\w+)\s+as\s+az'
        arviz_match = re.search(arviz_pattern, code)
        if arviz_match:
            found_packages.add(arviz_match.group(1))
        
        if not found_packages:
            return []
        
        # Use LLM to determine which are stdlib and what pip names are
        pip_package_names = self._resolve_package_names_with_llm(found_packages)
        
        # Check which packages are missing
        installed_set = self.package_manager.get_installed_packages()
        missing = []
        
        for import_name, pip_name in pip_package_names.items():
            if pip_name is None:  # Standard library, skip
                continue
            # Skip invalid/unknown (e.g. MCMC is from numpyro, not a package name)
            if not pip_name or pip_name.lower() in ('unknown', 'none', ''):
                continue
            
            # Check if installed (handle both hyphen and underscore variants)
            pkg_variants = [
                pip_name.lower(),
                pip_name.lower().replace('-', '_'),
                pip_name.lower().replace('_', '-')
            ]
            
            if not any(variant in installed_set for variant in pkg_variants):
                missing.append(pip_name)
        
        # Install missing packages
        for pkg in missing:
            if self._install_missing_package(pkg):
                installed_packages.append(pkg)
        
        return installed_packages
    
    def _resolve_package_names_with_llm(self, import_names: set) -> dict:
        """
        Use LLM to resolve import names to pip package names.
        
        Returns:
            Dict mapping import_name -> pip_package_name (or None if stdlib)
        """
        if not self.llm:
            # Fallback: assume import name = pip name (works for most packages)
            return {name: name for name in import_names}
        
        # Build prompt for LLM
        packages_list = sorted(import_names)
        prompt = f"""For each Python import name below, determine:
1. Is it a standard library module? (return "STDLIB")
2. If not, what is the pip package name? (e.g., sklearn -> scikit-learn, PIL -> Pillow)

Import names: {', '.join(packages_list)}

Return ONLY a JSON object mapping each import name to either "STDLIB" or the pip package name.
Example format:
{{
  "sys": "STDLIB",
  "sklearn": "scikit-learn",
  "pandas": "pandas",
  "arviz": "arviz"
}}

Output ONLY the JSON, no explanations:"""
        
        try:
            response = self.llm.reason(prompt, stream=False)
            
            # Parse JSON response
            import json
            # Clean response (remove markdown if present)
            response = response.strip()
            if response.startswith('```'):
                response = re.sub(r'```(?:json)?\s*', '', response)
                response = re.sub(r'```\s*$', '', response)
            
            result = json.loads(response)
            
            # Convert "STDLIB" to None
            resolved = {}
            for import_name in import_names:
                value = result.get(import_name, import_name)  # Default to import name if not found
                resolved[import_name] = None if value == "STDLIB" else value
            
            logger.info(f"  [LLM] Resolved {len(import_names)} packages: {resolved}")
            return resolved
            
        except Exception as e:
            logger.warning(f"  [LLM] Failed to resolve package names: {e}, using fallback")
            # Fallback: assume import name = pip name
            return {name: name for name in import_names}
        
