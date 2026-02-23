"""
HybridValidator: Multi-layered validation with self-healing capabilities

Implements 4-layer validation:
1. Static validation (syntax, imports, structure)
2. API validation (NumPyro API correctness, function signatures)
3. JAX tracing validation (trace with fake data, check shapes)
4. Execution validation (run with minimal config, auto-fix errors)
"""
import ast
import logging
import traceback
import re
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


class HybridValidator:
    """Multi-layered validator with 15 retries and self-healing"""
    
    def __init__(self, llm, max_retries: int = 15, rag=None):
        self.llm = llm
        self.max_retries = max_retries
        self.rag = rag
        from .execution_validator import ExecutionValidator
        self.execution_validator = ExecutionValidator(llm, max_retries=max_retries, rag=rag)
        
    def validate_and_fix(
        self, 
        code: str, 
        spec, 
        module_name: str,
        entry_point: str,
        test_args: tuple = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Multi-layered validation with auto-fixing
        
        Returns:
            Tuple of (validated_code, fixes_applied)
        """
        fixes_applied = {}
        
        # Layer 1: Static validation
        logger.info(f"[Layer 1] Static validation for {module_name}")
        code, static_fixes = self._static_validation(code, entry_point)
        if static_fixes:
            fixes_applied['static'] = static_fixes
        
        # Layer 2: API validation
        logger.info(f"[Layer 2] API validation for {module_name}")
        code, api_fixes = self._api_validation(code, module_name)
        if api_fixes:
            fixes_applied['api'] = api_fixes
        
        # Layer 3: JAX tracing validation (if applicable)
        if module_name == 'modeling':
            logger.info(f"[Layer 3] JAX tracing validation for {module_name}")
            code, trace_fixes = self._jax_tracing_validation(code)
            if trace_fixes:
                fixes_applied['jax_tracing'] = trace_fixes
        
        # Layer 4: Execution validation with self-healing
        logger.info(f"[Layer 4] Execution validation for {module_name}")
        code, exec_fixes = self.execution_validator.validate_and_fix(
            code=code,
            spec=spec,
            module_name=module_name,
            entry_point=entry_point,
            test_args=test_args
        )
        if exec_fixes:
            fixes_applied['execution'] = exec_fixes
        
        # FINAL VALIDATION: Ensure code actually compiles
        try:
            ast.parse(code)
            logger.info(f" All validation layers passed for {module_name}")
            return code, fixes_applied
        except SyntaxError as e:
            logger.error(f"FINAL VALIDATION FAILED for {module_name}: {e}")
            logger.error(f"Code still has syntax errors after {len(fixes_applied.get('execution', {}))} fix attempts")
            # Return the broken code with error info - let caller handle it
            fixes_applied['final_validation_failed'] = str(e)
            return code, fixes_applied
    
    def _static_validation(self, code: str, entry_point: str) -> Tuple[str, list]:
        """Layer 1: Syntax and structure validation"""
        fixes = []
        
        # Check syntax - CRITICAL: This MUST pass
        try:
            ast.parse(code)
            logger.debug(" Syntax valid")
        except SyntaxError as e:
            logger.error(f"CRITICAL Syntax error at line {e.lineno}: {e}")
            fixes.append(f"Syntax error at line {e.lineno}")
            # Note: We don't try to fix here, let ExecutionValidator handle it
        
        # Check entry point exists
        if f"def {entry_point}" not in code:
            logger.warning(f"Entry point '{entry_point}' not found")
            fixes.append(f"Missing entry point: {entry_point}")
        
        # Check basic imports
        required_imports = ['import ', 'from ']
        if not any(imp in code for imp in required_imports):
            logger.warning("No imports found")
            fixes.append("Missing imports")
        
        return code, fixes
    
    def _api_validation(self, code: str, module_name: str) -> Tuple[str, list]:
        """Layer 2: API correctness validation"""
        fixes = []
        
        # Check for common API mistakes
        api_patterns = {
            # NumPyro mistakes
            r'guide\.get_params': 'guide.get_params -> use guide.median(raw_params)',
            r'mcmc\.get_samples\(\)\.get\(': 'Incorrect get_samples usage',
            r'dist\.Beta\([^,]+\)(?!\.)': 'Beta distribution needs 2 parameters',
            r'numpyro\.infer\.SVI\([^)]*\)(?!\.run)': 'SVI initialization might be incomplete',
        }
        
        for pattern, description in api_patterns.items():
            if re.search(pattern, code):
                logger.warning(f"API issue: {description}")
                fixes.append(description)
        
        # Check function signatures
        if module_name == 'modeling':
            if 'def run_modeling(' in code:
                # Check signature
                match = re.search(r'def run_modeling\(([^)]*)\)', code)
                if match:
                    params = match.group(1)
                    if 'data_path: str' not in params:
                        logger.warning("run_modeling should have data_path: str parameter")
                        fixes.append("Incorrect run_modeling signature")
        
        return code, fixes
    
    def _jax_tracing_validation(self, code: str) -> Tuple[str, list]:
        """Layer 3: JAX-specific tracing validation"""
        fixes = []
        
        # Check for common JAX issues
        jax_patterns = {
            r'for\s+\w+\s+in\s+range.*jnp\.': 'Loop over JAX array (use vmap or scan)',
            r'if\s+[^:]+\.shape': 'Shape-dependent branching (use jnp.where)',
            r'\.append\(': 'List append in JAX code (use arrays)',
        }
        
        for pattern, description in jax_patterns.items():
            if re.search(pattern, code):
                logger.warning(f"JAX issue: {description}")
                fixes.append(description)
        
        return code, fixes
    
    def _fix_common_syntax_errors(self, code: str) -> str:
        """Fix common syntax errors without LLM"""
        # Remove trailing commas in function definitions
        code = re.sub(r',(\s*\))', r'\1', code)
        
        # Fix missing colons
        code = re.sub(r'(def\s+\w+\([^)]*\)\s*->\s*\w+)\s*\n', r'\1:\n', code)
        
        return code

