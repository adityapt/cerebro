"""
Base class for specialized MMM agents.

Each agent in the pipeline:
- Has ONE focused responsibility
- Generates 20-30 lines of code max
- Executes and validates its code
- Passes results to next agent
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from cerebro.utils.logging import get_logger

logger = get_logger(__name__)


class BaseAgent:
    """
    Base class for MMM pipeline agents.
    
    Each agent:
    1. Receives state from previous agent
    2. Generates focused code for its task (20-30 lines)
    3. Executes code with validation
    4. Returns updated state for next agent
    """
    
    def __init__(
        self,
        llm,
        agent_name: str,
        use_code_judge: bool = True,
        use_results_judge: bool = True
    ):
        """
        Initialize agent.
        
        Args:
            llm: Language model backend
            agent_name: Name of this agent (for logging)
            use_code_judge: Whether to validate/iterate on generated code
            use_results_judge: Whether to validate business logic of results
        """
        self.llm = llm
        self.agent_name = agent_name
        self.use_code_judge = use_code_judge
        self.use_results_judge = use_results_judge
        
        # Initialize code judge (validates if code will execute)
        if use_code_judge:
            from cerebro.llm.code_judge import CodeJudge
            self.judge = CodeJudge(llm)
        else:
            self.judge = None
        
        # Initialize results judge (validates business logic for ALL agents)
        if use_results_judge:
            from cerebro.llm.results_judge import ResultsJudge
            self.results_judge = ResultsJudge(llm)
            logger.info(f"  + ResultsJudge enabled (validates outputs)")
        else:
            self.results_judge = None
        
        logger.info(f"Initialized {agent_name}")
    
    def get_task_prompt(self, state: Dict[str, Any]) -> str:
        """
        Generate prompt for this agent's specific task.
        Should be overridden by subclasses.
        
        Args:
            state: Current pipeline state
            
        Returns:
            Prompt string for code generation
        """
        raise NotImplementedError(f"{self.agent_name} must implement get_task_prompt()")
    
    def get_execution_namespace(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare namespace with variables for code execution.
        
        Args:
            state: Current pipeline state
            
        Returns:
            Dict of variables to inject into code execution
        """
        # Base namespace with common imports and utilities
        namespace = {
            'pd': pd,
            'np': np,
            'state': state,  # Give code access to full state
        }
        
        # Add state variables directly to namespace for convenience
        for key, value in state.items():
            if isinstance(key, str) and key.isidentifier():
                namespace[key] = value
        
        # Also add profile variables to namespace (media_channels, kpi_col, etc.)
        if 'profile' in state:
            profile = state['profile']
            if isinstance(profile, dict):
                for key, value in profile.items():
                    if isinstance(key, str) and key.isidentifier():
                        namespace[key] = value
        
        return namespace
    
    def generate_code(self, state: Dict[str, Any]) -> str:
        """
        Generate code for this agent's task.
        
        Args:
            state: Current pipeline state
            
        Returns:
            Generated Python code
        """
        from cerebro.llm.hybrid_backend import HybridBackend
        logger.info(f"[{self.agent_name}] Generating code...")
        
        # Check if using Hybrid backend with reflection
        if isinstance(self.llm, HybridBackend):
            logger.info(f"[{self.agent_name}] Using Hybrid backend (DeepSeek + DeepAnalyze approach)")
            
            # Get profile for context
            profile = state.get('profile', {})
            
            # Build data_info from state
            data_info = {
                'media_channels': profile.get('media_channels', []),
                'kpi_col': profile.get('kpi_col', 'target'),
                'date_col': profile.get('date_col', 'date'),
                'shape': profile.get('shape', (0, 0))
            }
            
            # Get agent-specific task
            task = self.get_task_description(state)
            
            # Use Hybrid backend's reflection-based code generation
            code = self.llm.generate_code_with_reflection(
                task=task,
                data_info=data_info,
                agent_name=self.agent_name
            )
        else:
            # Standard code generation for other backends
            prompt = self.get_task_prompt(state)
            code = self.llm.reason(prompt=prompt)
            
            # Extract code from markdown if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
        
        logger.debug(f"[{self.agent_name}] Generated {len(code)} chars of code")
        return code
    
    def get_task_description(self, state: Dict[str, Any]) -> str:
        """
        Get task prompt for Hybrid backend.
        
        Returns the FULL prompt with all context, not just a one-line summary.
        This ensures the LLM has complete information to make autonomous decisions.
        
        Args:
            state: Current pipeline state
            
        Returns:
            Full task prompt with all context
        """
        # Return FULL prompt to preserve context!
        # The LLM needs to understand the problem domain, not just a one-liner.
        return self.get_task_prompt(state)
    
    def validate_and_fix_code(self, code: str, state: Dict[str, Any]) -> str:
        """
        Validate code and fix common issues.
        
        Validates that all column references in generated code actually exist in the data.
        
        Args:
            code: Generated code
            state: Current pipeline state
            
        Returns:
            Fixed/validated code
        """
        import re
        from difflib import get_close_matches
        
        fixes_applied = []
        
        # Fix 0: Validate column references exist in actual data
        # Get actual columns from data
        if 'data' in state:
            actual_columns = state['data'].columns.tolist()
        else:
            # Fallback: use profile columns
            profile = state.get('profile', {})
            actual_columns = []
            if 'media_channels' in profile:
                actual_columns.extend(profile['media_channels'])
            if 'kpi_col' in profile and profile['kpi_col']:
                actual_columns.append(profile['kpi_col'])
            if 'date_col' in profile and profile['date_col']:
                actual_columns.append(profile['date_col'])
        
        if actual_columns:
            # Extract all column references from code
            # Pattern: data['col'], data["col"], df['col'], df["col"], 'col' in data.columns, etc.
            patterns = [
                r"data\[(['\"])(.+?)\1\]",      # data['col']
                r"df\[(['\"])(.+?)\1\]",        # df['col']
                r"\.loc\[:,\s*(['\"])(.+?)\1",  # .loc[:, 'col']
                r"columns\s*==\s*(['\"])(.+?)\1",  # columns == 'col'
                r"for\s+\w+\s+in\s+\[(['\"])(.+?)\1",  # for x in ['col']
            ]
            
            column_refs = set()
            for pattern in patterns:
                matches = re.findall(pattern, code)
                for match in matches:
                    # match is a tuple: (quote_char, column_name)
                    col_name = match[1] if len(match) > 1 else match[0]
                    column_refs.add(col_name)
            
            # Validate each reference
            for col_ref in column_refs:
                if col_ref not in actual_columns:
                    # Column doesn't exist! Find closest match
                    matches = get_close_matches(col_ref, actual_columns, n=1, cutoff=0.6)
                    
                    if matches:
                        actual_col = matches[0]
                        logger.warning(f"Column '{col_ref}' not in data, replacing with '{actual_col}'")
                        
                        # Replace all occurrences
                        code = code.replace(f"'{col_ref}'", f"'{actual_col}'")
                        code = code.replace(f'"{col_ref}"', f'"{actual_col}"')
                        
                        fixes_applied.append(f"Fixed invalid column: {col_ref[:30]}... → {actual_col[:30]}...")
                    else:
                        logger.error(f"Column '{col_ref}' not found in data and no close match!")
        
        # Also check if code iterates over media_channels list
        profile = state.get('profile', {})
        media_channels = profile.get('media_channels', [])
        if media_channels and 'for' in code.lower() and 'media_channels' in code:
            # Code iterates over media_channels - this is good!
            # But warn if it also has hardcoded column names
            if any(f"'{ch}'" in code or f'"{ch}"' in code for ch in media_channels[:3]):
                logger.warning("Code has both media_channels loop AND hardcoded columns - may need review")
        
        # Fix 1: Remove CSV loading
        if "pd.read_csv" in code or "read_csv" in code:
            code = "\n".join([
                line for line in code.split("\n")
                if "read_csv" not in line
            ])
            fixes_applied.append("Removed CSV loading")
        
        # Fix 2: Remove return statements (script context)
        if "\nreturn " in code or code.strip().startswith("return "):
            code = "\n".join([
                f"# {line}" if line.strip().startswith("return ") else line
                for line in code.split("\n")
            ])
            fixes_applied.append("Commented out return statements")
        
        # Fix 3: Check for results variable
        if "results" not in code and "output" not in code:
            logger.warning(f"[{self.agent_name}] Code doesn't set output variable")
        
        if fixes_applied:
            logger.info(f"[{self.agent_name}] Applied {len(fixes_applied)} fixes: {fixes_applied}")
        else:
            logger.info(f"[{self.agent_name}] No fixes needed")
        
        return code
    
    def execute_code(
        self,
        code: str,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute generated code with validation.
        
        Args:
            code: Python code to execute
            state: Current pipeline state
            
        Returns:
            Results dict from code execution
        """
        logger.info(f"[{self.agent_name}] Executing code...")
        
        namespace = self.get_execution_namespace(state)
        
        try:
            exec(code, namespace)
            
            # Extract results - look for common output variable names
            # Prioritize based on agent type
            if self.agent_name == 'ModelBuilding':
                output_vars = ['model_results', 'results', 'output', 'transformed_data']
            elif self.agent_name == 'FeatureEngineering':
                output_vars = ['transformed_data', 'results', 'output']
            else:
                output_vars = ['results', 'output', 'transformed_data', 'model_results']
            
            results = None
            
            for var_name in output_vars:
                if var_name in namespace:
                    results = namespace[var_name]
                    logger.info(f"[{self.agent_name}] Found output in '{var_name}'")
                    break
            
            if results is None:
                logger.warning(f"[{self.agent_name}] No standard output variable found")
                # Return any new variables added to namespace
                results = {
                    k: v for k, v in namespace.items()
                    if k not in self.get_execution_namespace(state)
                    and not k.startswith('_')
                    and k not in ['pd', 'np', 'state']
                }
            
            return results if isinstance(results, dict) else {'output': results}
            
        except Exception as e:
            logger.error(f"[{self.agent_name}] Execution failed: {e}")
            return {'error': str(e)}
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution: Generate → Validate → Execute → Return
        
        Args:
            state: Current pipeline state
            
        Returns:
            Updated state with this agent's results
        """
        logger.info(f"[{self.agent_name}] Starting...")
        
        # Generate code
        code = self.generate_code(state)
        
        # Validate and fix
        code = self.validate_and_fix_code(code, state)
        
        # Iterate with judge if enabled
        max_iterations = 5 if self.use_code_judge else 1  # Increased from 3 to 5 for complex bugs
        
        for iteration in range(max_iterations):
            # Validate with judge on first attempt
            if self.use_code_judge and self.judge and iteration == 0:
                logger.info(f"[{self.agent_name}] Validating with Code Judge...")
                is_valid, code, critique = self.judge.validate_code(
                    code=code,
                    task=f"{self.agent_name} task",
                    data_info=state
                )
                if not is_valid:
                    logger.warning(f"[{self.agent_name}] Validation issues: {critique}")
            
            # Execute
            logger.info(f"[{self.agent_name}] Executing (attempt {iteration + 1}/{max_iterations})...")
            results = self.execute_code(code, state)
            
            # Check execution success
            if 'error' not in results:
                logger.info(f"[{self.agent_name}] ✓ Execution succeeded!")
                
                # Validate business logic (agent-specific validation)
                if self.use_results_judge and self.results_judge:
                    logger.info(f"[{self.agent_name}] Validating output quality...")
                    
                    # Call appropriate validation method based on agent type
                    if self.agent_name == "FeatureEngineering":
                        # Validate transformed data quality
                        transformed_data = results.get('output', results)
                        validation = self.results_judge.validate_feature_engineering_results(
                            original_data=state.get('data', {}),
                            transformed_data=transformed_data,
                            context=f"Agent: {self.agent_name}"
                        )
                        is_valid_key = 'is_valid'
                    elif self.agent_name == "ModelBuilding":
                        # Validate model results
                        model_results = results.get('output', results)
                        validation = self.results_judge.validate_mmm_results(
                            results=model_results,
                            data_info=state.get('profile', {}),
                            context=f"Agent: {self.agent_name}"
                        )
                        is_valid_key = 'makes_business_sense'
                    else:
                        # Skip validation for other agents
                        validation = {'is_valid': True}
                        is_valid_key = 'is_valid'
                    
                    if not validation.get(is_valid_key, True):
                        logger.warning(f"[{self.agent_name}] Business logic violations detected!")
                        logger.warning(f"  Issues: {validation.get('issues', [])}")
                        logger.warning(f"  Root causes: {validation.get('root_causes', [])}")
                        logger.warning(f"  Severity: {validation.get('severity', 'unknown')}")
                        
                        # Only fix if severity is high and we have iterations left
                        if validation.get('severity') in ['high', 'medium'] and iteration < max_iterations - 1:
                            logger.info(f"[{self.agent_name}] Regenerating code to fix output quality issues...")
                            
                            # Use appropriate fix method based on agent type
                            if self.agent_name == "ModelBuilding":
                                output_results = results.get('output', results)
                                fixed_code = self.results_judge.fix_mmm_results(
                                    code=code,
                                    results=output_results,
                                    issues=validation.get('issues', []),
                                    root_causes=validation.get('root_causes', []),
                                    recommendations=validation.get('recommendations', []),
                                    data_info=state.get('profile', {})
                                )
                                code = self.validate_and_fix_code(fixed_code, state)
                            else:
                                # For Feature Engineering and other agents, regenerate from scratch
                                # (We don't have a specific fix method for them yet)
                                logger.warning(f"[{self.agent_name}] Regenerating code from scratch...")
                                code = self.generate_code(state)  # Regenerate
                                code = self.validate_and_fix_code(code, state)
                            
                            logger.info(f"[{self.agent_name}] Code regenerated with quality fixes")
                            continue  # Re-execute with fixed code
                        else:
                            logger.warning(f"[{self.agent_name}] Continuing despite quality issues")
                    else:
                        logger.info(f"[{self.agent_name}] Output quality validation passed!")
                
                return results
            
            if not self.use_code_judge or iteration == max_iterations - 1:
                break
            
            # Ask judge to fix (AGENTIC: provides full execution context)
            logger.warning(f"[{self.agent_name}] Failed: {results['error']}")
            logger.info(f"[{self.agent_name}] Asking judge to fix...")
            
            # Get namespace keys for context
            namespace = self.get_execution_namespace(state)
            namespace_keys = [k for k in namespace.keys() if not k.startswith('_')]
            
            # Use Judge's agentic fix method
            fixed_code = self.judge.fix_execution_error(
                code=code,
                error_message=results['error'],
                task=f"{self.agent_name} task",
                data_info=state.get('profile', {}),
                namespace_keys=namespace_keys
            )
            
            code = self.validate_and_fix_code(fixed_code, state)
            logger.info(f"[{self.agent_name}] Code updated for retry")
        
        # Failed after all iterations
        logger.error(f"[{self.agent_name}] ✗ Failed after {max_iterations} attempts")
        return {'error': f'{self.agent_name} failed after {max_iterations} attempts'}

