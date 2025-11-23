"""
Tree of Thought reasoning for Cerebro.

Since we use local LLMs (no API costs), we can afford to explore
multiple reasoning paths for better quality results.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

from cerebro.llm.ollama_backend import OllamaBackend
from cerebro.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ThoughtNode:
    """A node in the tree of thought."""
    content: str
    path: List[str]
    score: float
    depth: int
    is_terminal: bool = False
    code: Optional[str] = None
    results: Optional[Dict] = None


class TreeOfThought:
    """
    Tree of Thought reasoning for complex data science tasks.
    
    Explores multiple approaches:
    - Different statistical methods
    - Alternative assumptions
    - Various data transformations
    
    Evaluates each path and selects best.
    """
    
    def __init__(
        self,
        llm: OllamaBackend,
        breadth: int = 3,  # How many alternatives to explore at each level
        depth: int = 3,    # How deep to search
        verbose: bool = True
    ):
        """
        Initialize Tree of Thought.
        
        Args:
            llm: LLM backend
            breadth: Number of alternatives per level (3-5 recommended)
            depth: Search depth (3-4 recommended)
            verbose: Print reasoning process
        """
        self.llm = llm
        self.breadth = breadth
        self.depth = depth
        self.verbose = verbose
    
    def solve(
        self,
        task: str,
        data_info: Dict[str, Any],
        context: Optional[str] = None
    ) -> ThoughtNode:
        """
        Solve task using Tree of Thought.
        
        Args:
            task: Data science task to solve
            data_info: Information about data
            context: Optional context
            
        Returns:
            Best solution found
        """
        
        logger.info(f"Tree of Thought: Exploring {self.breadth} approaches")
        
        # Root: Generate initial approaches
        root_thoughts = self._generate_approaches(task, data_info, context)
        
        # Build tree
        all_nodes = []
        for i, thought in enumerate(root_thoughts):
            logger.debug(f"Exploring Approach {i+1}: {thought[:100]}...")
            
            # Expand this path
            terminal_nodes = self._expand_path(
                thought=thought,
                task=task,
                data_info=data_info,
                current_depth=0
            )
            all_nodes.extend(terminal_nodes)
        
        # Select best solution
        best_node = max(all_nodes, key=lambda n: n.score)
        
        logger.info(f"Best approach selected (score: {best_node.score:.3f})")
        
        return best_node
    
    def _generate_approaches(
        self,
        task: str,
        data_info: Dict,
        context: Optional[str]
    ) -> List[str]:
        """
        Generate multiple initial approaches to the problem.
        
        Returns list of different approaches to try.
        """
        
        prompt = f"""You are a data science expert. Generate {self.breadth} different 
approaches to solve this task. Each should use different methods or assumptions.

Task: {task}

Data: 
- Shape: {data_info.get('shape')}
- Columns: {data_info.get('columns', [])[:10]}
- Numeric: {len(data_info.get('numeric_cols', []))}
- Categorical: {len(data_info.get('categorical_cols', []))}

Context: {context or 'None'}

Generate {self.breadth} distinct approaches:

1. [Approach name]: Brief description of method
2. [Approach name]: Brief description of method
3. [Approach name]: Brief description of method

Examples for A/B test:
1. Frequentist t-test: Classical statistical test with normality assumptions
2. Bootstrap: Non-parametric approach, no distribution assumptions
3. Bayesian: Probabilistic approach with credible intervals

Be specific about which methods to use.
"""
        
        response = self.llm.reason(prompt)
        
        # Parse approaches (simplified)
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        # Safe parsing: check line is not empty before accessing first char
        approaches = [line for line in lines if line and line[0].isdigit()][:self.breadth]
        
        if len(approaches) < self.breadth:
            # Fallback: at least have one approach
            logger.warning(f"Could not parse {self.breadth} approaches from LLM response. Using entire response.")
            approaches = [response]
        
        return approaches
    
    def _expand_path(
        self,
        thought: str,
        task: str,
        data_info: Dict,
        current_depth: int
    ) -> List[ThoughtNode]:
        """
        Expand a thought path to terminal nodes.
        
        Returns all terminal nodes from this path.
        """
        
        if current_depth >= self.depth:
            # Terminal: Generate code and return
            node = self._create_terminal_node(thought, task, data_info, current_depth)
            return [node]
        
        # Generate next steps for this approach
        next_thoughts = self._generate_next_steps(thought, task, data_info)
        
        # Recursively expand each
        all_terminals = []
        for next_thought in next_thoughts:
            terminals = self._expand_path(
                thought=next_thought,
                task=task,
                data_info=data_info,
                current_depth=current_depth + 1
            )
            all_terminals.extend(terminals)
        
        return all_terminals
    
    def _generate_next_steps(
        self,
        current_thought: str,
        task: str,
        data_info: Dict
    ) -> List[str]:
        """Generate next reasoning steps from current thought."""
        
        prompt = f"""Continue this data science reasoning. Generate the next step.

Task: {task}
Current approach: {current_thought}

What's the next specific step? Consider:
- Data validation checks needed
- Assumptions to verify
- Specific implementation details

Next step:"""
        
        response = self.llm.reason(prompt)
        
        # For now, return single next step (can expand to multiple)
        return [current_thought + "\n→ " + response]
    
    def _create_terminal_node(
        self,
        thought: str,
        task: str,
        data_info: Dict,
        depth: int
    ) -> ThoughtNode:
        """
        Create terminal node with code and evaluation.
        
        Returns ThoughtNode with score.
        """
        
        # Generate code based on this reasoning path
        code = self.llm.generate_code(
            task=task,
            data_info=data_info,
            requirements=[thought]
        )
        
        # Score this approach
        score = self._score_approach(thought, code, task, data_info)
        
        return ThoughtNode(
            content=thought,
            path=[thought],  # Simplified, should track full path
            score=score,
            depth=depth,
            is_terminal=True,
            code=code,
            results=None  # Will be filled after execution
        )
    
    def _score_approach(
        self,
        thought: str,
        code: str,
        task: str,
        data_info: Dict
    ) -> float:
        """
        Score an approach for quality.
        
        Uses LLM to evaluate based on:
        - Appropriateness for data
        - Statistical rigor
        - Robustness
        - Interpretability
        
        Returns score 0.0-1.0
        """
        
        prompt = f"""Evaluate this data science approach.

Task: {task}
Data: {data_info}

Approach:
{thought}

Generated Code:
```python
{code}
```

Score this approach (0.0-1.0) based on:
1. Appropriateness for the data characteristics
2. Statistical rigor and validity
3. Robustness to assumptions
4. Interpretability of results

Respond with JSON:
{{
    "score": 0.85,
    "reasoning": "This approach is appropriate because...",
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1"]
}}
"""
        
        response = self.llm.reason(prompt)
        
        # Parse score
        try:
            evaluation = json.loads(response)
            score = evaluation.get('score', 0.5)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse evaluation JSON: {e}")
            logger.debug(f"Response was: {response[:200]}")
            # Fallback: moderate score
            score = 0.5
        except Exception as e:
            logger.error(f"Unexpected error parsing evaluation: {e}")
            score = 0.5
        
        return float(score)


class GraphOfThought:
    """
    Graph of Thought for complex interdependent problems.
    
    Use for:
    - Marketing Mix Modeling (channel interactions)
    - Causal inference with complex confounding
    - Multi-objective optimization
    """
    
    def __init__(
        self,
        llm: OllamaBackend,
        verbose: bool = True
    ):
        self.llm = llm
        self.verbose = verbose
    
    def solve(
        self,
        task: str,
        data_info: Dict[str, Any],
        dependencies: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Solve task with Graph of Thought.
        
        Models dependencies between decisions.
        
        Args:
            task: Complex task with interdependencies
            data_info: Data information
            dependencies: Known dependencies between variables
            
        Returns:
            Solution considering all dependencies
        """
        
        logger.info("Graph of Thought: Modeling interdependencies")
        
        # For MVP, defer to ToT
        # Full GoT implementation in future
        tot = TreeOfThought(self.llm, verbose=self.verbose)
        return tot.solve(task, data_info)

