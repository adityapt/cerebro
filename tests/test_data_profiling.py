"""
Tests for data profiling functionality.
"""

import pytest
import pandas as pd
import numpy as np


def test_profile_data():
    """Test data profiling works correctly."""
    from cerebro.core.orchestrator import Cerebro
    
    # Create test data
    data = pd.DataFrame({
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [1.5, 2.5, 3.5, 4.5, 5.5],
        'categorical': ['A', 'B', 'A', 'C', 'B'],
        'mixed': [1, 'two', 3, 'four', 5]
    })
    
    cerebro = Cerebro.__new__(Cerebro)
    profile = cerebro._profile_data(data)
    
    # Check profile structure
    assert 'shape' in profile
    assert profile['shape'] == (5, 4)
    
    assert 'columns' in profile
    assert len(profile['columns']) == 4
    
    assert 'dtypes' in profile
    assert len(profile['dtypes']) == 4
    
    assert 'numeric_cols' in profile
    assert 'numeric1' in profile['numeric_cols']
    assert 'numeric2' in profile['numeric_cols']
    
    assert 'categorical_cols' in profile
    assert 'categorical' in profile['categorical_cols']
    
    assert 'missing' in profile


def test_profile_experiment_data():
    """Test experiment-specific data profiling."""
    from cerebro.core.orchestrator import Cerebro
    
    # Create experiment data
    data = pd.DataFrame({
        'user_id': range(100),
        'group': ['control'] * 50 + ['treatment'] * 50,
        'metric': np.random.randn(100)
    })
    
    cerebro = Cerebro.__new__(Cerebro)
    profile = cerebro._profile_experiment_data(data, 'group', 'metric')
    
    # Check experiment-specific fields (nested in 'experiment' dict)
    assert 'experiment' in profile
    exp = profile['experiment']
    
    assert 'groups' in exp
    assert 'control' in exp['groups']
    assert 'treatment' in exp['groups']
    
    assert 'group_sizes' in exp
    assert exp['group_sizes']['control'] == 50
    assert exp['group_sizes']['treatment'] == 50
    
    assert 'metric' in exp
    assert exp['metric'] == 'metric'

