"""
Execution-based code validation with feedback loop
Tests generated code on sample data and provides concrete error feedback
"""
import tempfile
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ExecutionValidator:
    """
    Validates generated code by actually executing it on sample data.
    Much more reliable than abstract code review.
    """
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
    
    def validate_and_fix(
        self,
        code: str,
        data_path: str,
        agent,
        context: str = ""
    ) -> Tuple[str, bool, str]:
        """
        Execute code on sample data and fix errors iteratively.
        
        Args:
            code: Generated Python code
            data_path: Path to data file
            agent: Agent to call for fixes
            context: Context about what this code does
            
        Returns:
            (fixed_code, success, error_message)
        """
        logger.info("="*80)
        logger.info("EXECUTION-BASED VALIDATION")
        logger.info("="*80)
        
        current_code = code
        
        for attempt in range(self.max_retries):
            logger.info(f"\nAttempt {attempt + 1}/{self.max_retries}")
            
            # Try to execute on sample data
            success, error_msg = self._execute_on_sample(current_code, data_path)
            
            if success:
                logger.info(f"✓ Code executed successfully on sample data!")
                return current_code, True, ""
            
            # Failed - give feedback to agent
            logger.warning(f"✗ Execution failed: {error_msg[:200]}...")
            
            if attempt < self.max_retries - 1:
                logger.info(f"Asking agent to fix the error...")
                current_code = self._ask_agent_to_fix(
                    current_code, error_msg, agent, context
                )
            else:
                logger.error("Max retries reached. Code still has errors.")
                return current_code, False, error_msg
        
        return current_code, False, error_msg
    
    def _execute_on_sample(
        self,
        code: str,
        data_path: str,
        sample_size: int = 100
    ) -> Tuple[bool, str]:
        """
        Execute code on a small sample of data to check for errors.
        
        Returns:
            (success, error_message)
        """
        # Create a sample data file
        sample_data_path = self._create_sample_data(data_path, sample_size)
        
        # Modify code to use sample data and reduce iterations
        sample_code = self._modify_for_sample(code, sample_data_path)
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(sample_code)
            temp_file = f.name
        
        try:
            # Execute in subprocess with timeout
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout for sample
            )
            
            if result.returncode == 0:
                return True, ""
            else:
                # Extract useful error message
                error_msg = self._extract_error(result.stderr, result.stdout)
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            return False, "Execution timed out after 30 seconds"
        except Exception as e:
            return False, f"Execution error: {str(e)}"
        finally:
            # Cleanup
            Path(temp_file).unlink(missing_ok=True)
            Path(sample_data_path).unlink(missing_ok=True)
    
    def _create_sample_data(self, data_path: str, sample_size: int) -> str:
        """Create a small sample of the data for testing"""
        import pandas as pd
        
        df = pd.read_csv(data_path)
        sample_df = df.head(sample_size)
        
        sample_path = tempfile.mktemp(suffix='.csv')
        sample_df.to_csv(sample_path, index=False)
        
        return sample_path
    
    def _modify_for_sample(self, code: str, sample_data_path: str) -> str:
        """
        Modify code to:
        1. Use sample data path
        2. Reduce iterations for speed
        3. Add error reporting
        """
        modified = code
        
        # Replace data path references
        modified = modified.replace(
            "pd.read_csv('data.csv')",
            f"pd.read_csv('{sample_data_path}')"
        )
        modified = modified.replace(
            'pd.read_csv("data.csv")',
            f'pd.read_csv("{sample_data_path}")'
        )
        
        # Reduce SVI iterations for speed (if present)
        modified = modified.replace(
            "num_steps=50000",
            "num_steps=100"
        )
        modified = modified.replace(
            "num_steps=10000",
            "num_steps=100"
        )
        
        # Add error reporting wrapper
        wrapper = f'''
import sys
import traceback

try:
{self._indent_code(modified, "    ")}
except Exception as e:
    print("ERROR:", str(e), file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''
        return wrapper
    
    def _indent_code(self, code: str, indent: str) -> str:
        """Add indentation to code"""
        return '\n'.join(indent + line if line.strip() else line 
                        for line in code.split('\n'))
    
    def _extract_error(self, stderr: str, stdout: str) -> str:
        """
        Extract the most useful error information from stderr/stdout.
        Focus on the actual error, not full stack trace.
        """
        if not stderr:
            return stdout[-500:] if stdout else "Unknown error"
        
        lines = stderr.split('\n')
        
        # Find the actual error (usually last few lines)
        error_lines = []
        for line in reversed(lines):
            if line.strip():
                error_lines.insert(0, line)
                if len(error_lines) >= 10:  # Get last 10 relevant lines
                    break
        
        # Extract key information
        error_msg = '\n'.join(error_lines)
        
        # Keep it concise but informative
        if len(error_msg) > 1000:
            error_msg = error_msg[-1000:]
        
        return error_msg
    
    def _ask_agent_to_fix(
        self,
        code: str,
        error_msg: str,
        agent,
        context: str
    ) -> str:
        """
        Ask the agent to fix the code based on the execution error.
        
        This is where the magic happens - concrete error feedback!
        """
        fix_prompt = f"""The following code has an execution error:

CONTEXT: {context}

ERROR:
{error_msg}

ORIGINAL CODE:
{code}

Fix this error and regenerate the complete code. 

CRITICAL:
1. The error above is from ACTUAL EXECUTION - fix the exact issue
2. Do NOT change working parts - only fix what's broken
3. Output ONLY valid Python code, no explanations
4. Keep all imports and structure intact

FIXED CODE:"""

        logger.info("Asking agent to fix the error...")
        
        # Stream the fix
        fixed_code = ""
        for token in agent.llm.reason(fix_prompt, stream=True):
            print(token, end="", flush=True)
            fixed_code += token
        
        print("\n")
        
        # Clean up the response
        fixed_code = fixed_code.strip()
        if fixed_code.startswith("```python"):
            fixed_code = fixed_code[len("```python"):].strip()
        if fixed_code.endswith("```"):
            fixed_code = fixed_code[:-len("```")].strip()
        
        return fixed_code


def validate_generated_code(
    code: str,
    data_path: str,
    agent,
    context: str = "Generated MMM pipeline"
) -> Tuple[str, bool]:
    """
    Convenience function to validate and fix generated code.
    
    Returns:
        (final_code, success)
    """
    validator = ExecutionValidator(max_retries=3)
    final_code, success, error = validator.validate_and_fix(
        code, data_path, agent, context
    )
    
    if not success:
        logger.error(f"Code validation failed after 3 attempts: {error}")
    
    return final_code, success

