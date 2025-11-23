"""
Safety tests for code execution.
"""

import pytest
import pandas as pd
import numpy as np


def test_dangerous_code_rejection():
    """Test that dangerous code patterns are rejected."""
    from cerebro.core.orchestrator import Cerebro
    
    # Mock LLM backend for testing without Ollama
    class MockLLM:
        def __init__(self, model):
            pass
        
        def _check_ollama_connection(self):
            pass
        
        def _ensure_model_available(self):
            pass
    
    cerebro = Cerebro.__new__(Cerebro)
    cerebro.verbose = False
    cerebro.use_tree_of_thought = False
    cerebro.use_code_judge = False
    
    # Test dangerous patterns
    dangerous_codes = [
        "import os\nos.system('rm -rf /')",
        "import subprocess\nsubprocess.run(['rm', '-rf', '/'])",
        "eval('malicious code')",
        "exec('malicious code')",
        "open('/etc/passwd', 'r')",
        "__import__('os').system('ls')",
    ]
    
    test_data = pd.DataFrame({'a': [1, 2, 3]})
    
    for code in dangerous_codes:
        result = cerebro._execute_code(code, {'data': test_data})
        assert 'error' in result, f"Should reject dangerous code: {code[:30]}"
        assert 'rejected' in result['error'].lower(), f"Error should mention rejection: {result['error']}"


def test_safe_code_execution():
    """Test that safe code executes successfully."""
    from cerebro.core.orchestrator import Cerebro
    
    cerebro = Cerebro.__new__(Cerebro)
    cerebro.verbose = False
    
    # Test safe code
    safe_code = """
import pandas as pd
import numpy as np

result = data['a'].mean()
results = {'mean': result}
"""
    
    test_data = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
    
    result = cerebro._execute_code(safe_code, {'data': test_data})
    assert 'error' not in result or result.get('error') is None
    assert 'mean' in result
    assert result['mean'] == 3.0


def test_code_without_results_variable():
    """Test code execution when 'results' variable is not defined."""
    from cerebro.core.orchestrator import Cerebro
    
    cerebro = Cerebro.__new__(Cerebro)
    cerebro.verbose = False
    
    # Code that creates variables but no 'results'
    code = """
import numpy as np
mean_val = np.mean([1, 2, 3, 4, 5])
sum_val = np.sum([1, 2, 3, 4, 5])
"""
    
    result = cerebro._execute_code(code, {})
    assert 'mean_val' in result
    assert result['mean_val'] == 3.0
    assert 'sum_val' in result
    assert result['sum_val'] == 15

