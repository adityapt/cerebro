"""
Results Judge for validating business logic.

Unlike CodeJudge (which validates if code will execute),
ResultsJudge validates if RESULTS make business sense.
"""

from typing import Dict, Any, Optional
import json

from cerebro.utils.logging import get_logger

logger = get_logger(__name__)


class ResultsJudge:
    """
    LLM-based results validator for business logic.
    
    Validates results from ANY agent:
    - FeatureEngineering: Is transformed data valid? (not flat, no NaNs)
    - ModelBuilding: Do coefficients make business sense? (positive, reasonable)
    - PostProcessing: Are ROI/attribution realistic? (sum to 100%, positive)
    
    This is AGENTIC validation - uses LLM to understand domain constraints.
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def validate_feature_engineering_results(
        self,
        original_data: Dict[str, Any],
        transformed_data: Dict[str, Any],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate Feature Engineering results (AGENTIC).
        
        Args:
            original_data: Original channel data (for comparison)
            transformed_data: Transformed channel data
            context: Optional additional context
            
        Returns:
            Dict with:
            - is_valid: bool
            - issues: list of problems
            - root_causes: why issues exist
            - recommendations: how to fix
        """
        
        # Extract statistics for validation
        import pandas as pd
        import numpy as np
        import types
        
        # Helper to make data JSON-safe
        def make_json_safe(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return f"<{type(obj).__name__}>"
            elif isinstance(obj, types.ModuleType):
                return f"<module '{obj.__name__}'>"
            elif callable(obj):
                return f"<callable>"
            elif isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_safe(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            else:
                return obj
        
        if isinstance(transformed_data, pd.DataFrame):
            stats = {
                'shape': transformed_data.shape,
                'columns': list(transformed_data.columns),
                'stats_per_column': {}
            }
            
            for col in transformed_data.columns:
                vals = transformed_data[col]
                stats['stats_per_column'][col] = {
                    'min': float(vals.min()),
                    'max': float(vals.max()),
                    'mean': float(vals.mean()),
                    'std': float(vals.std()),
                    'has_nan': bool(vals.isna().any()),
                    'has_inf': bool(np.isinf(vals).any()),
                    'is_constant': bool(vals.std() < 0.01)  # Flat if std < 0.01
                }
        else:
            # Clean transformed_data if it's a dict with non-serializable objects
            stats = make_json_safe(transformed_data)
        
        prompt = f"""You are a PRINCIPAL DATA SCIENTIST reviewing feature engineering output.

Your standards are HIGH. You conduct DATA QUALITY REVIEW and reject bad transformations.

Transformed Data Statistics:
{json.dumps(stats, indent=2)}

{f"Additional Context: {context}" if context else ""}

DATA QUALITY CHECKLIST (line-by-line inspection):

1. VARIATION CHECK (CRITICAL!)
   - Is std > 0.01 for each column? (If not → FLAT OUTPUT = BUG!)
   - Does data have meaningful variation? (not constant)
   - Red flags: std < 0.01, all values identical, only minor noise
   
2. NUMERICAL STABILITY
   - Any NaN values? (REJECT - division by zero or invalid math)
   - Any Inf values? (REJECT - overflow or unstable computation)
   - Any extreme outliers? (values > 1e6 or < -1e6)
   
3. TRANSFORMATION VALIDITY
   - Does output differ from input? (not just copying)
   - Reasonable range? (adstock ~similar, saturation 0-1 or compressed)
   - Pattern makes sense? (smoothing for adstock, compression for saturation)
   
4. BUSINESS LOGIC
   - Will this data work for regression? (needs variation!)
   - Would coefficients be learnable? (not if flat!)
   - Is transformation doing what it should? (carryover/saturation)

REJECTION CRITERIA (be STRICT!):
- std < 0.01 (flat output = transformation bug)
- Any NaN or Inf values
- All columns identical (no variation)
- Values exploded (>1e6) or collapsed (all near zero)

If ANY rejection criterion met → is_valid = FALSE

Respond in JSON:
{{
    "is_valid": true or false (FALSE if ANY rejection criterion met),
    "issues": ["EVERY problem found - be thorough!"],
    "root_causes": ["deep analysis of WHY issues exist"],
    "recommendations": ["specific fixes needed"],
    "severity": "low|medium|high",
    "data_quality_notes": ["specific findings per column"]
}}

Conduct DATA QUALITY REVIEW:"""
        
        # Use appropriate backend method
        if hasattr(self.llm, 'reason'):
            response = self.llm.reason(prompt)
        elif hasattr(self.llm, 'chat'):
            response = self.llm.chat([{"role": "user", "content": prompt}])
        else:
            response = self.llm.generate(prompt)
        
        # Parse response
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            validation = json.loads(json_str)
            
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Failed to parse feature engineering validation JSON: {e}")
            logger.debug(f"Response was: {response[:300]}")
            
            # Check for obvious issues in stats
            issues = []
            for col, col_stats in stats.get('stats_per_column', {}).items():
                if col_stats.get('is_constant'):
                    issues.append(f"{col} is flat (std={col_stats['std']:.6f})")
                if col_stats.get('has_nan'):
                    issues.append(f"{col} has NaN values")
                if col_stats.get('has_inf'):
                    issues.append(f"{col} has Inf values")
            
            validation = {
                "is_valid": len(issues) == 0,
                "issues": issues if issues else [],
                "root_causes": ["Could not parse full validation"],
                "recommendations": [],
                "severity": "high" if issues else "unknown"
            }
        
        return validation
    
    def validate_mmm_results(
        self,
        results: Dict[str, Any],
        data_info: Dict[str, Any],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate MMM results for business logic (AGENTIC).
        
        Args:
            results: Model results (coefficients, baseline, fit metrics)
            data_info: Context about the data
            context: Optional additional context
            
        Returns:
            Dict with:
            - is_valid: bool
            - issues: list of business logic issues
            - recommendations: what to do about them
        """
        
        # Helper to make data JSON-safe
        import pandas as pd
        import numpy as np
        import types
        
        def make_json_safe(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return f"<{type(obj).__name__}>"
            elif isinstance(obj, types.ModuleType):
                return f"<module '{obj.__name__}'>"
            elif callable(obj):
                return f"<callable>"
            elif isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_safe(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            else:
                return obj
        
        results_safe = make_json_safe(results)
        data_info_safe = make_json_safe(data_info)
        
        prompt = f"""You are a PRINCIPAL DATA SCIENTIST conducting final MODEL REVIEW for production.

Your standards are HIGH. You check if model results make business sense before deployment.

Model Results:
{json.dumps(results_safe, indent=2)}

Data Context:
{json.dumps(data_info_safe, indent=2)}

{f"Additional Context: {context}" if context else ""}

MODEL REVIEW CHECKLIST (strict validation):

1. COEFFICIENT SIGNS (CRITICAL!)
   - Are ALL coefficients positive? (spending more → more sales)
   - Red flags: Any negative coefficient (indicates multicollinearity/overfitting/bug)
   - Business rule: In MMM, negative coefficients are INVALID (except competitor brand)
   
2. BASELINE VALIDATION
   - Is baseline positive? (MUST be > 0)
   - Is baseline substantial? (should be 30-80% of avg sales)
   - Red flag: Baseline near zero (model claims all sales from media = wrong!)
   
3. MODEL FIT SANITY
   - Is R² reasonable? (0.6-0.95 is good)
   - Red flags: R² < 0.5 (model weak), R² > 0.99 (overfitting!), R² = 1.0 (perfect fit = bug!)
   - Is MAPE reasonable? (5-20% is good, >30% is bad)
   
4. STATISTICAL RED FLAGS
   - Check for multicollinearity (if VIF provided)
   - Check for heteroscedasticity (if test provided)
   - Check for autocorrelation (if Durbin-Watson provided)

5. BUSINESS LOGIC
   - Do results make intuitive sense?
   - Are coefficient magnitudes reasonable?
   - Would this model be trusted for decision-making?

REJECTION CRITERIA (be STRICT!):
- ANY negative coefficient (unless justified)
- Baseline <= 0 or > 95% of sales
- R² = 1.0 (perfect fit = overfitting)
- R² < 0.3 (useless model)

If ANY rejection criterion met → makes_business_sense = FALSE

Respond in JSON:
{{
    "makes_business_sense": true or false (FALSE if ANY rejection criterion met),
    "issues": ["EVERY business logic violation found"],
    "root_causes": ["deep analysis of WHY violations exist"],
    "recommendations": ["specific fixes - be precise!"],
    "severity": "low|medium|high",
    "coefficient_check": ["status of each coefficient"]
}}

Conduct MODEL REVIEW:"""
        
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
            
            validation = json.loads(json_str)
            
            # Map to expected format
            if 'makes_business_sense' in validation:
                validation['is_valid'] = validation['makes_business_sense']
            
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Failed to parse results validation JSON: {e}")
            logger.debug(f"Response was: {response[:300]}")
            
            # Try to infer from text
            response_lower = response.lower()
            makes_sense = 'makes sense' in response_lower or 'valid' in response_lower or 'reasonable' in response_lower
            
            validation = {
                "is_valid": makes_sense,
                "makes_business_sense": makes_sense,
                "issues": [] if makes_sense else ["Could not parse full validation"],
                "root_causes": [response[:200]] if not makes_sense else [],
                "recommendations": [],
                "severity": "unknown"
            }
        
        return validation
    
    def fix_mmm_results(
        self,
        code: str,
        results: Dict[str, Any],
        issues: list,
        root_causes: list,
        recommendations: list,
        data_info: Dict[str, Any]
    ) -> str:
        """
        Fix code to address business logic issues (AGENTIC).
        
        Args:
            code: Original code that produced bad results
            results: The problematic results
            issues: Business logic violations
            root_causes: Why issues exist
            recommendations: How to fix
            data_info: Data context
            
        Returns:
            Fixed code
        """
        
        prompt = f"""You are fixing MMM code that produces results violating business constraints.

Original code:
```python
{code}
```

Results produced:
{json.dumps(results, indent=2)}

Business logic violations:
{chr(10).join(f"- {issue}" for issue in issues)}

Root causes:
{chr(10).join(f"- {cause}" for cause in root_causes)}

Recommendations:
{chr(10).join(f"- {rec}" for rec in recommendations)}

Data context:
{json.dumps(data_info, indent=2)}

Your objectives:
- Understand why the model produces invalid results
- Research the best way to fix the business logic violations
- Generate corrected code that will produce valid results
- Consider adding:
  * Regularization (Ridge) if multicollinearity is the issue
  * Non-negative constraints if needed
  * Better feature engineering if data is insufficient
  * Diagnostic checks (VIF, condition number)

Generate the fixed code (just code, no explanations):
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

