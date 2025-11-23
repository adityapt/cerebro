"""
LLM Judge for validating generated code.

Implements self-critique pattern for code quality.
"""

from typing import Dict, List, Optional, Tuple
import json

from cerebro.llm.ollama_backend import OllamaBackend
from cerebro.utils.logging import get_logger

logger = get_logger(__name__)


class CodeJudge:
    """
    LLM-based code validator using self-critique.
    
    Checks generated code for:
    - Correctness
    - Safety
    - Efficiency
    - Best practices
    """
    
    def __init__(self, llm: OllamaBackend):
        self.llm = llm
    
    def validate_code(
        self,
        code: str,
        task: str,
        data_info: Dict,
        max_attempts: int = 3
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Validate generated code with LLM judge.
        
        Args:
            code: Generated code to validate
            task: What the code is supposed to do
            data_info: Information about the data
            max_attempts: Max refinement attempts
            
        Returns:
            (is_valid, final_code, critique)
        """
        
        current_code = code
        
        for attempt in range(max_attempts):
            # Judge the code
            critique = self._critique_code(current_code, task, data_info)
            
            # Check if code passes
            if critique['is_valid']:
                return True, current_code, None
            
            # If not, get improvement suggestions
            if attempt < max_attempts - 1:
                current_code = self._improve_code(
                    current_code,
                    critique['issues'],
                    task,
                    data_info
                )
        
        # Failed all attempts
        return False, current_code, critique['issues']
    
    def _critique_code(
        self,
        code: str,
        task: str,
        data_info: Dict
    ) -> Dict:
        """
        Have LLM critique the code (STRICT: Principal DS-level peer review).
        
        Returns dict with:
        - is_valid: bool
        - issues: list of problems
        - root_cause: explanation
        """
        
        prompt = f"""You are a PRINCIPAL DATA SCIENTIST conducting a rigorous peer review.

Your standards are HIGH. You do line-by-line code review and catch:
- Syntax errors, missing imports, undefined variables
- Logic bugs (wrong formulas, incorrect implementations)
- Data quality issues (would this produce flat/invalid output?)
- Edge cases (NaNs, infinities, division by zero)
- Code smells (overwrites variables, ignores provided data)

Code to review:
```python
{code}
```

Context:
- Task: {task}
- Data available: {data_info}

LINE-BY-LINE REVIEW CHECKLIST:

1. IMPORTS & DEPENDENCIES
   - Are all imports present?
   - Are functions/modules correctly imported?

2. DATA USAGE
   - Does code use provided data variables (not create fake data)?
   - Does code overwrite provided variables?
   - Are column names used correctly?

3. LOGIC & CORRECTNESS
   - Are formulas implemented correctly?
   - Are transformations valid? (not using only first value, etc.)
   - Would output be flat/constant? (check for bugs like series[0])

4. NUMERICAL STABILITY
   - Any division by zero?
   - Any overflow/underflow risk?
   - NaN or Inf values possible?

5. OUTPUT VALIDITY
   - Will output variable be created?
   - Will output have correct shape/type?
   - Will output preserve data variation (not flat)?

Be STRICT. Reject code that:
- Has any syntax/execution errors
- Creates fake data instead of using provided data
- Would produce flat/constant output (transformation bugs)
- Has logic errors in formulas

Respond in JSON:
{{
    "will_execute": true or false,
    "issues": ["list of EVERY issue found (be thorough!)"],
    "root_cause": "deep analysis of why issues exist",
    "confidence": "high|medium|low",
    "line_by_line_notes": ["specific issues per line"]
}}

Conduct your peer review:"""
        
        # Use appropriate backend method
        if hasattr(self.llm, 'reason'):
            response = self.llm.reason(prompt)
        elif hasattr(self.llm, 'chat'):
            response = self.llm.chat([{"role": "user", "content": prompt}])
        else:
            response = self.llm.generate(prompt)
        
        # Parse response
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            critique = json.loads(json_str)
            
            # Map 'will_execute' to 'is_valid' for backward compatibility
            if 'will_execute' in critique:
                critique['is_valid'] = critique['will_execute']
            
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Failed to parse code critique JSON: {e}")
            logger.debug(f"Response was: {response[:300]}")
            
            # Try to infer from text response
            response_lower = response.lower()
            will_execute = 'will execute' in response_lower or 'looks good' in response_lower or 'valid' in response_lower
            
            critique = {
                "is_valid": will_execute,
                "will_execute": will_execute,
                "issues": [] if will_execute else ["Code may have issues (could not parse full analysis)"],
                "root_cause": response[:200] if not will_execute else "Code appears valid",
                "confidence": "low"
            }
        
        return critique
    
    def _improve_code(
        self,
        code: str,
        issues: List[str],
        task: str,
        data_info: Dict
    ) -> str:
        """
        Improve code based on critique (STRICT: Principal DS fixing ALL issues).
        
        Returns improved code.
        """
        
        prompt = f"""You are a PRINCIPAL DATA SCIENTIST fixing code that FAILED peer review.

Your peer review found CRITICAL ISSUES. Fix them ALL.

Original goal: {task}

Context:
- Data available: {data_info}

Code that was REJECTED:
```python
{code}
```

Issues found during line-by-line review:
{chr(10).join(f"- {issue}" for issue in issues)}

FIX REQUIREMENTS (ALL must be satisfied):

1. Address EVERY issue identified above
2. Use provided data variables (DO NOT create fake data!)
3. Implement formulas correctly (no bugs like series[0])
4. Ensure output preserves variation (not flat!)
5. Handle edge cases (NaN, Inf, zeros)
6. Add all necessary imports
7. Follow MMM best practices

CRITICAL CHECKS before submitting fix:
✓ Does code use 'data' variable from scope (not create new data)?
✓ Will transformations preserve variation (not produce flat output)?
✓ Are formulas implemented correctly?
✓ Are all imports present?

Generate the CORRECTED code that will PASS peer review (code only):
"""
        
        # Use appropriate backend method
        if hasattr(self.llm, 'reason'):
            response = self.llm.reason(prompt)
        elif hasattr(self.llm, 'chat'):
            response = self.llm.chat([{"role": "user", "content": prompt}])
        elif hasattr(self.llm, 'generate_code'):
            return self.llm.generate_code(
                task=task,
                data_info=data_info,
                requirements=[f"Fix: {issue}" for issue in issues]
            )
        else:
            response = self.llm.generate(prompt)
        
        # Extract code from response
        if "```python" in response:
            improved_code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            improved_code = response.split("```")[1].split("```")[0].strip()
        else:
            improved_code = response.strip()
        
        return improved_code
    
    def fix_execution_error(
        self,
        code: str,
        error_message: str,
        task: str,
        data_info: Dict,
        namespace_keys: Optional[List[str]] = None
    ) -> str:
        """
        Fix code that failed during execution (STRICT: Principal DS debugging).
        
        Args:
            code: Code that failed
            error_message: The error message
            task: What the code was supposed to do
            data_info: Data context
            namespace_keys: Variables available in execution namespace
            
        Returns:
            Fixed code
        """
        
        # Classify error type for better guidance
        is_import_error = 'ImportError' in error_message or 'ModuleNotFoundError' in error_message or 'cannot import' in error_message
        is_pymc_arviz_error = is_import_error and ('pymc' in error_message.lower() or 'arviz' in error_message.lower() or 'gaussian' in error_message.lower())
        
        # Build context-aware prompt
        if is_pymc_arviz_error:
            error_guidance = """
ERROR TYPE: ImportError (PyMC/ArviZ dependency conflict)

This is a DEPENDENCY ISSUE, not a code bug!

CRITICAL: DO NOT try to fix PyMC or ArviZ imports!
The environment has incompatible library versions.

SOLUTION: SWITCH to a different library entirely!

Your options (in order of preference):

1. NumPyro (RECOMMENDED for Bayesian MMM):
   ```python
   import numpyro
   import numpyro.distributions as dist
   from numpyro.infer import MCMC, NUTS
   import jax.numpy as jnp
   ```
   - No arviz dependency
   - JAX-accelerated (fast)
   - Full Bayesian inference with HalfNormal priors

2. scipy Ridge with bounds (RECOMMENDED for frequentist MMM):
   ```python
   from scipy.optimize import minimize
   import numpy as np
   ```
   - Always works (no complex dependencies)
   - Can enforce positive constraints with bounds
   - Simple and reliable

3. sklearn BayesianRidge (fallback):
   ```python
   from sklearn.linear_model import BayesianRidge
   ```
   - Approximate Bayesian inference
   - No dependency issues

DO NOT USE: PyMC, arviz, stan (dependency conflicts in this environment)

ACTION REQUIRED: Rewrite the code using NumPyro or scipy Ridge.
"""
        elif is_import_error:
            # Extract failed library
            failed_lib = "unknown"
            if "cannot import name" in error_message:
                try:
                    failed_lib = error_message.split("from")[1].split("(")[0].strip().strip("'\"")
                except:
                    pass
            
            error_guidance = f"""
ERROR TYPE: ImportError

The library/module cannot be imported: {failed_lib}

CRITICAL: This is a missing dependency, not a code bug!

SOLUTION: Use an alternative library that IS available!

Common alternatives:
- For numerical operations: numpy, scipy (always available)
- For ML: sklearn (always available)
- For Bayesian: numpyro (if jax available), sklearn.linear_model.BayesianRidge
- For optimization: scipy.optimize (always available)

DO NOT: Try to fix the import or install the package
DO: Rewrite code using available libraries

ACTION REQUIRED: Identify the goal of the code and implement it using available libraries.
"""
        else:
            # Classify common logic errors and provide specific guidance
            error_lower = error_message.lower()
            
            if 'keyerror' in error_lower or 'not in index' in error_lower:
                # Pandas KeyError - trying to access non-existent column/key
                error_guidance = f"""
ERROR TYPE: KeyError (pandas/dict key doesn't exist)

Error message: {error_message}

ANALYSIS:
- You're trying to access a column or dictionary key that doesn't exist
- Common causes:
  1. Trying to access df[['column_name']] when column_name isn't in the DataFrame
  2. Trying to access dict['key'] when key doesn't exist in the dict
  3. Typo in column/key name

SPECIFIC GUIDANCE FOR THIS ERROR:

If error mentions 'intercept':
  → DO NOT access df[['intercept']] (intercept is not a DataFrame column!)
  → Use model_results['baseline'] or model_results['intercept'] (from dict, not DataFrame)
  → The intercept/baseline is a scalar value, not a column

If error mentions column names:
  → Check what columns actually exist (print df.columns)
  → Use exact column names from the data
  → Don't make up column names

ACTION REQUIRED:
1. Identify what key/column you're trying to access
2. Check if it exists in the data structure (DataFrame vs dict)
3. Use the correct access method (df['col'] vs dict['key'])
4. Fix the code to access the correct key/column
"""
            
            elif 'nameerror' in error_lower or 'is not defined' in error_lower:
                error_guidance = f"""
ERROR TYPE: NameError (variable not defined)

Error message: {error_message}

ANALYSIS:
- You're using a variable that doesn't exist in scope
- Common causes:
  1. Typo in variable name
  2. Variable not created yet (used before defined)
  3. Variable from previous code not available

SPECIFIC GUIDANCE:

Available variables in scope: {namespace_keys if namespace_keys else 'Check execution namespace'}

ACTION REQUIRED:
1. Check spelling of variable name
2. Ensure variable is defined before use
3. Check if variable is in the provided namespace
4. Don't create fake variables - use provided data
"""
            
            elif 'typeerror' in error_lower:
                error_guidance = f"""
ERROR TYPE: TypeError (wrong data type)

Error message: {error_message}

ANALYSIS:
- You're performing an operation on incompatible data types
- Common causes:
  1. Trying to do math on non-numeric types (str * int)
  2. Passing wrong type to function (list instead of array)
  3. Using incorrect indexing (array[list] instead of array[array])

ACTION REQUIRED:
1. Check data types of variables (print type(variable))
2. Convert to correct type (pd.Series, np.array, float, etc.)
3. Ensure operation is valid for the data type
"""
            
            elif 'valueerror' in error_lower:
                error_guidance = f"""
ERROR TYPE: ValueError (invalid value)

Error message: {error_message}

ANALYSIS:
- A function received a valid type but invalid value
- Common causes:
  1. NaN or Inf values in numeric operations
  2. Shape mismatch in array operations
  3. Invalid parameter value (negative where positive expected)

ACTION REQUIRED:
1. Check for NaN/Inf values (df.isna().sum(), np.isinf())
2. Check array shapes (print X.shape, y.shape)
3. Validate parameter ranges (ensure positive where needed)
4. Add data validation before operations
"""
            
            elif 'attributeerror' in error_lower:
                error_guidance = f"""
ERROR TYPE: AttributeError (method/attribute doesn't exist)

Error message: {error_message}

ANALYSIS:
- You're calling a method that doesn't exist on the object
- Common causes:
  1. Wrong object type (calling DataFrame method on array)
  2. Method name typo
  3. Library version incompatibility

ACTION REQUIRED:
1. Check object type (print type(object))
2. Verify method name (use dir(object) to see available methods)
3. Ensure object is correct type for the operation
"""
            
            else:
                # Generic runtime error
                error_guidance = f"""
ERROR TYPE: Runtime Error

Error message: {error_message}

ANALYSIS:
This is a logic bug in the code.

ACTION REQUIRED:
1. Read the error message carefully
2. Identify the exact line and operation that failed
3. Check data types, values, and structures
4. Fix the implementation
"""
        
        prompt = f"""You are a PRINCIPAL DATA SCIENTIST debugging a runtime failure.

This code CRASHED. Find the bug and fix it.

Original goal: {task}

Context:
- Data available: {data_info}
{f"- Variables available in scope: {namespace_keys}" if namespace_keys else ""}

Code that CRASHED:
```python
{code}
```

Runtime Error:
```
{error_message}
```

{error_guidance}

DEBUGGING STEPS:

1. READ THE ERROR MESSAGE CAREFULLY
   - What exactly failed?
   - What library/function/variable caused it?
   
2. IDENTIFY THE ROOT CAUSE
   - Is it an import error? → Switch libraries!
   - Is it a logic bug? → Fix the implementation!
   - Is it a data issue? → Check column names, types, values!
   
3. CHOOSE THE FIX STRATEGY
   - Import errors: Use different library (NumPyro or scipy Ridge)
   - Logic bugs: Correct the implementation
   - Data issues: Use correct variable/column names

FIX REQUIREMENTS:
✓ If ImportError: SWITCH to a working library (NumPyro or scipy Ridge)
✓ If logic bug: Fix the implementation
✓ Use provided data variables (don't create fake data!)
✓ Ensure code will execute successfully

Generate the FIXED code (code only, no explanations):
"""
        
        # Use appropriate backend method
        if hasattr(self.llm, 'reason'):
            response = self.llm.reason(prompt)
        elif hasattr(self.llm, 'chat'):
            response = self.llm.chat([{"role": "user", "content": prompt}])
        else:
            response = self.llm.generate(prompt)
        
        # Extract code from response
        if "```python" in response:
            fixed_code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            fixed_code = response.split("```")[1].split("```")[0].strip()
        else:
            fixed_code = response.strip()
        
        return fixed_code

