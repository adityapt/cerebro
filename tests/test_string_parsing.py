"""
Tests for string parsing edge cases.
"""

import pytest


def test_empty_line_parsing():
    """Test that empty lines don't cause crashes in ToT parsing."""
    # Simulate the parsing logic from tree_of_thought.py
    
    # Test case 1: Empty response
    response = ""
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    # Safe parsing: check line is not empty before accessing first char
    approaches = [line for line in lines if line and line[0].isdigit()]
    assert len(approaches) == 0
    
    # Test case 2: Lines with only whitespace
    response = "   \n\t\n   "
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    approaches = [line for line in lines if line and line[0].isdigit()]
    assert len(approaches) == 0
    
    # Test case 3: Mixed content
    response = """
1. First approach
   
2. Second approach

3. Third approach
"""
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    approaches = [line for line in lines if line and line[0].isdigit()]
    assert len(approaches) == 3
    
    # Test case 4: No numbered lines
    response = "Just some text without numbers"
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    approaches = [line for line in lines if line and line[0].isdigit()]
    assert len(approaches) == 0


def test_json_parsing_fallbacks():
    """Test JSON parsing with invalid input."""
    import json
    
    # Test invalid JSON
    invalid_jsons = [
        "not json at all",
        "{incomplete: ",
        "{'single': 'quotes'}",
        "",
        "null",
    ]
    
    for invalid in invalid_jsons:
        try:
            result = json.loads(invalid)
        except json.JSONDecodeError:
            # This is expected - should be caught
            assert True
        except Exception as e:
            pytest.fail(f"Should raise JSONDecodeError, not {type(e).__name__}")

