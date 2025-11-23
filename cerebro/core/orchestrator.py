"""
Main Orchestrator for Cerebro agentic system.

Implements ReAct pattern (Reason + Act) with multi-agent routing.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import logging
import json

from cerebro.llm.ollama_backend import OllamaBackend
from cerebro.llm.tree_of_thought import TreeOfThought, GraphOfThought
from cerebro.llm.code_judge import CodeJudge
from cerebro.utils.logging import get_logger

logger = get_logger(__name__)


class Cerebro:
    """
    Main orchestrator agent for autonomous data science.
    
    Routes tasks to specialist agents:
    - Experiment Agent: A/B testing, CUPED, power analysis
    - Causal Agent: Synthetic controls, DiD, matching
    - Forecast Agent: Time series, Prophet, ARIMA
    - MMM Agent: Marketing mix modeling
    
    Uses ReAct pattern:
    1. Reason: Analyze data and decide approach
    2. Act: Execute via specialist agent
    3. Observe: Check results
    4. Reflect: Improve if needed
    """
    
    def __init__(
        self,
        model: str = "deepseek-coder-v2:16b",  # DeepSeek 16B - smarter than Qwen 7B
        verbose: bool = True,
        max_iterations: int = 5,
        use_tree_of_thought: bool = False,  # Default to CoT (fast, ~2-3 min)
        use_code_judge: bool = True         # Default to validation (iterates on errors)
    ):
        """
        Initialize Cerebro orchestrator.
        
        Args:
            model: LLM model name (via Ollama)
            verbose: Print progress
            max_iterations: Max attempts for iterative improvement
            use_tree_of_thought: Use ToT for better quality (recommended)
            use_code_judge: Validate code before execution (recommended)
        """
        self.llm = OllamaBackend(model=model)
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.use_tree_of_thought = use_tree_of_thought
        self.use_code_judge = use_code_judge
        
        # Advanced reasoning (no API costs with local LLMs)
        if use_tree_of_thought:
            self.tot = TreeOfThought(self.llm, verbose=verbose)
            self.got = GraphOfThought(self.llm, verbose=verbose)
        
        # Code validation
        if use_code_judge:
            self.judge = CodeJudge(self.llm)
        
        # Initialize specialist agents (will implement later)
        self.experiment_agent = None
        self.causal_agent = None
        self.forecast_agent = None
        self.mmm_agent = None
        
        # Memory (will implement later)
        self.short_term_memory = []
        self.long_term_memory = None
        
        if self.verbose:
            reasoning_mode = "Tree of Thought" if use_tree_of_thought else "Chain of Thought"
            validation = "with code validation" if use_code_judge else ""
            logger.info(f"Cerebro initialized with {model}")
            logger.info(f"Reasoning: {reasoning_mode} {validation}")
            # Set debug level if verbose
            logging.getLogger("cerebro").setLevel(logging.DEBUG)
    
    def analyze(
        self,
        data: pd.DataFrame,
        question: str,
        context: Optional[str] = None
    ) -> "AnalysisResult":
        """
        Main entry point: Autonomous analysis from natural language.
        
        Args:
            data: Input DataFrame
            question: What to analyze (natural language)
            context: Optional business context
            
        Returns:
            AnalysisResult with findings, code, and interpretation
        
        Example:
            >>> result = brain.analyze(
            ...     data=df,
            ...     question="Analyze this A/B test and tell me if it's significant"
            ... )
        """
        logger.info(f"Analyzing: {question}")
        
        # Step 1: Profile data
        data_profile = self._profile_data(data)
        
        # Step 2: Reason about task
        task_analysis = self._analyze_task(question, data_profile, context)
        
        # Step 3: Route to specialist agent
        result = self._route_and_execute(task_analysis, data, question, context)
        
        return result
    
    def analyze_experiment(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        metric: str,
        pre_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> "ExperimentResult":
        """
        Specialized method for A/B test analysis.
        
        Args:
            data: Experiment data
            treatment_col: Column indicating treatment/control
            metric: Metric to analyze
            pre_data: Optional pre-experiment data for CUPED
            
        Returns:
            ExperimentResult with significance, effect size, etc.
        """
        logger.info("Analyzing experiment")
        
        # Profile data
        data_profile = self._profile_experiment_data(data, treatment_col, metric)
        
        # Check for CUPED opportunity
        cuped_opportunity = pre_data is not None
        
        # Generate analysis code
        code = self._generate_experiment_code(
            data_profile=data_profile,
            treatment_col=treatment_col,
            metric=metric,
            use_cuped=cuped_opportunity
        )
        
        # Execute code
        results = self._execute_code(code, {'data': data, 'pre_data': pre_data})
        
        # Interpret results
        interpretation = self.llm.interpret_results(
            task="A/B test analysis",
            results=results
        )
        
        return ExperimentResult(
            significant=results.get('significant', False),
            p_value=results.get('p_value'),
            effect_size=results.get('effect_size'),
            confidence_interval=results.get('confidence_interval'),
            interpretation=interpretation,
            code=code,
            raw_results=results
        )
    
    def estimate_causal_impact(
        self,
        data: pd.DataFrame,
        treatment_units: List[str],
        outcome: str,
        treatment_start: str,
        **kwargs
    ) -> "CausalResult":
        """
        Estimate causal impact using synthetic control or DiD.
        
        Args:
            data: Panel data (units × time)
            treatment_units: Units that received treatment
            outcome: Outcome variable
            treatment_start: When treatment started
            
        Returns:
            CausalResult with treatment effect estimate
        """
        logger.info("Estimating causal impact")
        
        # This will be implemented with Causal Agent
        raise NotImplementedError("Causal Agent coming in Week 2")
    
    def forecast(
        self,
        data: pd.DataFrame,
        target: str,
        horizon: int,
        freq: str = 'D',
        **kwargs
    ) -> "ForecastResult":
        """
        Time series forecasting.
        
        Args:
            data: Time series data
            target: Variable to forecast
            horizon: How many periods ahead
            freq: Frequency ('D', 'W', 'M', etc.)
            
        Returns:
            ForecastResult with predictions and intervals
        """
        logger.info("Forecasting")
        
        # This will be implemented with Forecast Agent
        raise NotImplementedError("Forecast Agent coming in Week 3")
    
    def build_mmm(
        self,
        data: pd.DataFrame,
        date_col: Optional[str] = None,
        kpi_col: Optional[str] = None,
        media_channels: Optional[List[str]] = None,
        control_vars: Optional[List[str]] = None,
        approach: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Build Marketing Mix Model using agentic approach.
        
        The MMM Agent will:
        1. Auto-detect columns if not specified
        2. Profile the data structure
        3. Generate appropriate MMM code
        4. Calculate ROI and attribution
        5. Provide optimization recommendations
        
        Args:
            data: Time series data with media spend and KPI
            date_col: Date column name (auto-detected if None)
            kpi_col: KPI/sales column name (auto-detected if None)
            media_channels: List of media channel columns (auto-detected if None)
            control_vars: Optional control variables
            approach: 'simple', 'bayesian', 'neural', or 'auto'
            
        Returns:
            Dict with ROI, attribution, model fit, and recommendations
        """
        logger.info("Building MMM using MMM Agent")
        
        # Import MMM Agent
        from cerebro.agents.mmm_agent import MMMAgent
        
        # Initialize MMM Agent
        mmm_agent = MMMAgent(
            llm=self.llm,
            use_tree_of_thought=self.use_tree_of_thought,
            use_code_judge=self.use_code_judge
        )
        
        # Build MMM
        results = mmm_agent.build_mmm(
            data=data,
            date_col=date_col,
            kpi_col=kpi_col,
            media_channels=media_channels,
            control_vars=control_vars,
            approach=approach
        )
        
        return results
    
    def _profile_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Profile data to understand characteristics."""
        profile = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'missing': data.isnull().sum().to_dict(),
            'numeric_cols': list(data.select_dtypes(include=['number']).columns),
            'categorical_cols': list(data.select_dtypes(include=['object', 'category']).columns)
        }
        
        # Basic statistics for numeric columns
        if profile['numeric_cols']:
            profile['summary_stats'] = data[profile['numeric_cols']].describe().to_dict()
        
        return profile
    
    def _profile_experiment_data(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        metric: str
    ) -> Dict[str, Any]:
        """Profile experiment-specific data."""
        profile = self._profile_data(data)
        
        # Experiment-specific info
        profile['experiment'] = {
            'treatment_col': treatment_col,
            'metric': metric,
            'groups': data[treatment_col].unique().tolist(),
            'group_sizes': data[treatment_col].value_counts().to_dict(),
            'metric_by_group': data.groupby(treatment_col)[metric].agg(['mean', 'std', 'count']).to_dict()
        }
        
        return profile
    
    def _analyze_task(
        self,
        question: str,
        data_profile: Dict[str, Any],
        context: Optional[str]
    ) -> Dict[str, Any]:
        """
        Reason about what task to perform.
        
        Uses LLM to classify task and plan approach.
        """
        prompt = f"""Analyze this data science task and determine the approach.

Question: {question}

Data Profile:
- Shape: {data_profile['shape']}
- Columns: {', '.join(data_profile['columns'][:10])}...
- Numeric columns: {len(data_profile['numeric_cols'])}
- Categorical columns: {len(data_profile['categorical_cols'])}

Context: {context or 'None'}

Determine:
1. Task type (experiment, causal_inference, forecasting, mmm, exploratory)
2. Which specialist agent should handle this
3. Key considerations or potential issues

Respond with JSON:
{{
    "task_type": "...",
    "agent": "experiment|causal|forecast|mmm",
    "reasoning": "...",
    "considerations": ["..."]
}}
"""
        
        response = self.llm.reason(prompt)
        
        # Parse response (simplified for MVP)
        try:
            analysis = json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse task analysis JSON: {e}")
            logger.debug(f"Response was: {response[:200]}")
            # Fallback: simple keyword matching
            question_lower = question.lower()
            if any(k in question_lower for k in ['a/b', 'ab test', 'experiment']):
                analysis = {'task_type': 'experiment', 'agent': 'experiment'}
            elif any(k in question_lower for k in ['causal', 'synthetic control', 'did']):
                analysis = {'task_type': 'causal_inference', 'agent': 'causal'}
            elif any(k in question_lower for k in ['forecast', 'predict', 'time series']):
                analysis = {'task_type': 'forecasting', 'agent': 'forecast'}
            elif any(k in question_lower for k in ['mmm', 'marketing mix', 'attribution']):
                analysis = {'task_type': 'mmm', 'agent': 'mmm'}
            else:
                analysis = {'task_type': 'exploratory', 'agent': 'experiment'}
        except Exception as e:
            logger.error(f"Unexpected error parsing task analysis: {e}")
            analysis = {'task_type': 'exploratory', 'agent': 'experiment'}  # Safe default
        
        return analysis
    
    def _route_and_execute(
        self,
        task_analysis: Dict[str, Any],
        data: pd.DataFrame,
        question: str,
        context: Optional[str]
    ) -> "AnalysisResult":
        """Route task to appropriate specialist agent."""
        agent_type = task_analysis.get('agent', 'experiment')
        
        logger.debug(f"Routing to {agent_type} agent")
        
        # For MVP, implement basic experiment analysis
        if agent_type == 'experiment':
            # Generate and execute code
            code = self._generate_analysis_code(question, data, task_analysis)
            results = self._execute_code(code, {'data': data})
            interpretation = self.llm.interpret_results(
                task=question,
                results=results,
                business_context=context
            )
            
            return AnalysisResult(
                findings=interpretation,
                code=code,
                results=results,
                task_type=task_analysis.get('task_type')
            )
        else:
            raise NotImplementedError(f"{agent_type} agent coming in future weeks")
    
    def _generate_experiment_code(
        self,
        data_profile: Dict[str, Any],
        treatment_col: str,
        metric: str,
        use_cuped: bool
    ) -> str:
        """Generate code for experiment analysis using ToT if enabled."""
        
        task = f"A/B test analysis: compare '{metric}' between groups in '{treatment_col}'"
        
        if use_cuped:
            task += " with CUPED variance reduction"
        
        # Use Tree of Thought if enabled (better quality)
        if self.use_tree_of_thought:
            logger.debug("Using Tree of Thought to explore multiple approaches")
            
            best_node = self.tot.solve(
                task=task,
                data_info=data_profile,
                context="A/B testing experiment"
            )
            code = best_node.code
        else:
            # Fallback to simple code generation
            requirements = [
                f"Analyze experiment with treatment column '{treatment_col}' and metric '{metric}'",
                "Run statistical significance test",
                "Calculate effect size and confidence intervals",
                "Check for multiple testing if needed"
            ]
            
            if use_cuped:
                requirements.append("Apply CUPED variance reduction using pre-experiment data")
            
            code = self.llm.generate_code(
                task=task,
                data_info=data_profile,
                requirements=requirements
            )
        
        # Validate code with LLM judge if enabled
        if self.use_code_judge:
            logger.debug("Validating generated code")
            
            is_valid, validated_code, critique = self.judge.validate_code(
                code=code,
                task=task,
                data_info=data_profile
            )
            
            if is_valid:
                logger.debug("Code validated successfully")
                code = validated_code
            else:
                logger.warning(f"Code has issues: {critique}")
                logger.warning("Using best available version")
                code = validated_code
        
        return code
    
    def _generate_analysis_code(
        self,
        question: str,
        data: pd.DataFrame,
        task_analysis: Dict[str, Any]
    ) -> str:
        """Generate analysis code based on question."""
        data_profile = self._profile_data(data)
        
        code = self.llm.generate_code(
            task=question,
            data_info=data_profile,
            requirements=[
                "Provide complete analysis",
                "Return results as dictionary",
                "Include relevant statistics and visualizations"
            ]
        )
        
        return code
    
    def _execute_code(
        self,
        code: str,
        variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute generated code with safety checks.
        
        WARNING: Uses exec() which has security implications.
        For production, consider RestrictedPython or containerization.
        
        Args:
            code: Python code to execute
            variables: Variables to inject (e.g., {'data': df})
            
        Returns:
            Dictionary of results
        """
        # Safety checks for dangerous patterns
        dangerous_patterns = [
            ('import os', 'filesystem access'),
            ('import sys', 'system access'),
            ('import subprocess', 'subprocess execution'),
            ('__import__', 'dynamic imports'),
            ('eval(', 'eval execution'),
            ('exec(', 'nested exec'),
            ('open(', 'file operations'),
            ('compile(', 'code compilation'),
            ('globals()', 'global scope access'),
            ('locals()', 'local scope access'),
            ('__builtins__', 'builtin access'),
        ]
        
        code_lower = code.lower()
        for pattern, description in dangerous_patterns:
            if pattern.lower() in code_lower:
                logger.error(f"SECURITY: Rejected code containing {description}: {pattern}")
                return {
                    'error': f'Code rejected: contains potentially dangerous pattern ({description})',
                    'code': code
                }
        
        # Create execution namespace
        namespace = variables.copy()
        
        # Add common imports
        import pandas as pd
        import numpy as np
        from scipy import stats
        import statsmodels.api as sm
        
        namespace.update({
            'pd': pd,
            'np': np,
            'stats': stats,
            'sm': sm
        })
        
        # Note: We can't fully block __builtins__ without breaking Python's import system
        # The pattern checks above provide first-line defense
        # For true sandboxing, use RestrictedPython or containers
        
        try:
            logger.debug("Executing generated code with safety checks")
            # Execute code
            exec(code, namespace)
            
            # Extract results (code should define 'results' variable)
            if 'results' in namespace:
                return namespace['results']
            else:
                # Try to infer results
                results = {
                    k: v for k, v in namespace.items()
                    if not k.startswith('_') and k not in variables and callable(v) == False
                }
                return results
        
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return {'error': str(e), 'code': code}


class AnalysisResult:
    """Container for analysis results."""
    
    def __init__(
        self,
        findings: str,
        code: str,
        results: Dict[str, Any],
        task_type: str
    ):
        self.findings = findings
        self.code = code
        self.results = results
        self.task_type = task_type
    
    def summary(self) -> str:
        """Get summary of findings."""
        return self.findings


class ExperimentResult:
    """Container for experiment analysis results."""
    
    def __init__(
        self,
        significant: bool,
        p_value: float,
        effect_size: float,
        confidence_interval: tuple,
        interpretation: str,
        code: str,
        raw_results: Dict[str, Any]
    ):
        self.significant = significant
        self.p_value = p_value
        self.effect_size = effect_size
        self.confidence_interval = confidence_interval
        self.interpretation = interpretation
        self.code = code
        self.raw_results = raw_results
    
    def summary(self) -> str:
        """Get summary of experiment results."""
        sig_text = "SIGNIFICANT" if self.significant else "NOT SIGNIFICANT"
        return f"""
Experiment Analysis Results:
- Significance: {sig_text} (p={self.p_value:.4f})
- Effect Size: {self.effect_size:.4f}
- 95% CI: [{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]

Interpretation:
{self.interpretation}
"""


class CausalResult:
    """Container for causal inference results."""
    pass


class ForecastResult:
    """Container for forecasting results."""
    pass


class MMMResult:
    """Container for MMM results."""
    pass

