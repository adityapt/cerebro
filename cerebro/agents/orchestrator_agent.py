"""
🎭 Orchestrator Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Coordinates all specialized agents to generate complete MMM pipeline.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import logging
from pathlib import Path
from cerebro.spec.schema import MMMSpec
from cerebro.llm import AutoBackend
from cerebro.agents.data_exploration_agent import DataExplorationAgent
from cerebro.agents.preprocessing_agent import PreprocessingAgent
from cerebro.agents.modeling_agent import ModelingAgent
from cerebro.agents.diagnostics_agent import DiagnosticsAgent
from cerebro.agents.optimization_agent import OptimizationAgent
from cerebro.agents.visualization_agent import VisualizationAgent
from cerebro.agents.code_cleaner_agent import CodeCleanerAgent

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Orchestrates multiple specialized agents to generate complete MMM pipeline.
    
    Architecture:
    ┌─────────────────────────────────────┐
    │      OrchestratorAgent              │
    └─────────┬───────────────────────────┘
              │
              ├─→ DataExplorationAgent   → EDA code (~200 lines)
              ├─→ PreprocessingAgent     → Cleaning code (~150 lines)
              ├─→ ModelingAgent          → Bayesian model (~400 lines)
              ├─→ DiagnosticsAgent       → Validation (~300 lines)
              ├─→ OptimizationAgent      → Budget (~200 lines)
              └─→ VisualizationAgent     → Plots (~250 lines)
                            ↓
              Final code: ~1500 lines of production MMM
    """
    
    def __init__(self, llm: AutoBackend, use_rag: bool = True):
        self.llm = llm
        self.use_rag = use_rag
        
        # Initialize code cleaner agent
        self.cleaner = CodeCleanerAgent(llm)
        
        # Initialize all specialist agents
        self.agents = {
            'exploration': DataExplorationAgent(llm, use_rag),
            'preprocessing': PreprocessingAgent(llm, use_rag),
            'modeling': ModelingAgent(llm, use_rag),
            'diagnostics': DiagnosticsAgent(llm, use_rag),
            'optimization': OptimizationAgent(llm, use_rag),
            'visualization': VisualizationAgent(llm, use_rag),
        }
    
    def generate_complete_pipeline(self, spec: MMMSpec, output_path: str = None, data_path: str = None) -> str:
        """
        Generate complete MMM pipeline by orchestrating all agents.
        
        Args:
            spec: MMMSpec object defining the model
            output_path: Optional path to save generated code
            data_path: Optional path to data file (for agent reference)
            
        Returns:
            complete_code: Full Python pipeline as string
        """
        logger.info("=" * 80)
        logger.info("🎭 ORCHESTRATOR: GENERATING COMPLETE MMM PIPELINE")
        logger.info("=" * 80)
        logger.info(f"\nModel: {spec.name}")
        logger.info(f"Channels: {len(spec.channels)}")
        logger.info(f"Backend: {spec.inference.backend}")
        logger.info(f"Agents: {len(self.agents)}")
        
        # Generate header
        header = self._generate_header(spec)
        
        # Call each agent in sequence
        modules = {}
        
        print("\n" + "=" * 80)
        print("MULTI-AGENT CODE GENERATION")
        print("=" * 80)
        
        modules['exploration'] = self.agents['exploration'].generate_eda_code(spec, data_path)
        modules['preprocessing'] = self.agents['preprocessing'].generate_preprocessing_code(spec, data_path)
        modules['modeling'] = self.agents['modeling'].generate_model_code(spec, data_path)
        modules['diagnostics'] = self.agents['diagnostics'].generate_diagnostics_code(spec, data_path)
        modules['optimization'] = self.agents['optimization'].generate_optimization_code(spec, data_path)
        modules['visualization'] = self.agents['visualization'].generate_visualization_code(spec, data_path)
        
        # Clean each module using CodeCleanerAgent
        logger.info("\n🧹 Cleaning generated code...")
        for name in modules:
            original_lines = len(modules[name].splitlines())
            modules[name] = self.cleaner.clean_generated_code(modules[name])
            cleaned_lines = len(modules[name].splitlines())
            logger.info(f"  {name}: {original_lines} → {cleaned_lines} lines")
        
        # Generate main execution
        main_execution = self._generate_main(spec)
        
        # Assemble complete code
        complete_code = self._assemble_code(header, modules, main_execution)
        
        # Print summary
        total_lines = len(complete_code.splitlines())
        logger.info("\n" + "=" * 80)
        logger.info("✅ COMPLETE PIPELINE GENERATED!")
        logger.info("=" * 80)
        logger.info(f"\nTotal lines: {total_lines}")
        for name, code in modules.items():
            logger.info(f"  {name:15s}: {len(code.splitlines())} lines")
        
        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(complete_code)
            logger.info(f"\n✓ Saved to: {output_path}")
        
        return complete_code
    
    def _generate_header(self, spec: MMMSpec) -> str:
        """Generate header with imports"""
        return f'''"""
🚀 AUTO-GENERATED PRODUCTION MMM PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model: {spec.name}
Channels: {len(spec.channels)}
Backend: {spec.inference.backend}
Generated by: Cerebro Multi-Agent System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide, Predictive
from numpyro.optim import Adam

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from statsmodels.tsa.stattools import acf

# Optional: ArviZ for diagnostics
try:
    import arviz as az
    HAS_ARVIZ = True
except ImportError:
    print("⚠️  ArviZ not available - some diagnostics will be skipped")
    HAS_ARVIZ = False

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
jax.config.update('jax_platform_name', 'cpu')

print("=" * 80)
print("🎯 PRODUCTION MMM PIPELINE")
print("=" * 80)
print(f"Model: {spec.name}")
print(f"Channels: {len(spec.channels)}")
print(f"Backend: {spec.inference.backend}")
print("=" * 80)

'''
    
    def _generate_main(self, spec: MMMSpec) -> str:
        """Generate main execution code"""
        channel_names = [ch.name for ch in spec.channels]
        control_names = spec.controls if spec.controls else []
        data_path = spec.data_path if hasattr(spec, 'data_path') and spec.data_path else 'data.csv'
        
        return f'''

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # Load data
    print("\\nLoading data from: {data_path}")
    data_df = pd.read_csv('{data_path}')
    print(f"✓ Loaded {{len(data_df)}} rows, {{len(data_df.columns)}} columns")
    
    # Define columns
    outcome_col = '{spec.outcome}'
    channel_cols = {channel_names}
    control_cols = {control_names}
    date_col = '{spec.date_column if hasattr(spec, "date_column") else None}'
    
    # Step 1: Explore data
    exploration_results = explore_data(data_df, outcome_col, channel_cols, date_col)
    
    # Step 2: Preprocess
    data_df_processed, preprocessing_report = preprocess_data(
        data_df, outcome_col, channel_cols, control_cols, date_col
    )
    
    # Step 3: Prepare for modeling
    print("\\n" + "=" * 80)
    print("🎯 MODEL TRAINING")
    print("=" * 80)
    
    data = {{ch: data_df[ch].values for ch in channel_cols}}
    if control_cols:
        data.update({{ctrl: data_df[ctrl].values for ctrl in control_cols}})
    
    outcome = data_df[outcome_col].values
    
    print(f"\\nData shape: {{len(outcome)}} observations")
    print(f"Channels: {{len(channel_cols)}}")
    print(f"Controls: {{len(control_cols)}}")
    
    # Step 4: Train model
    posterior_samples, losses, svi, svi_state = run_svi(data, outcome)
    
    # Step 5: Diagnostics
    diagnostics = diagnose_model(posterior_samples, losses, data, outcome, svi_state)
    
    # Step 6: Budget optimization
    current_spend = {{ch: data_df[ch].mean() for ch in channel_cols[:10]}}
    total_budget = sum(current_spend.values())
    
    channel_names_clean = [ch.replace('.', '_').replace('-', '_') for ch in channel_cols[:10]]
    optimal_allocation, expected_lift = optimize_budget(
        posterior_samples, channel_names_clean, current_spend, total_budget
    )
    
    # Step 7: Visualize
    predictions = diagnostics.get('predictions', outcome)  # Fallback
    viz_results = visualize_results(
        data_df, outcome, predictions, posterior_samples, channel_cols, date_col
    )
    
    print("\\n" + "=" * 80)
    print("✅ COMPLETE MMM PIPELINE EXECUTED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\\nR²: {{diagnostics.get('r2', 'N/A')}}")
    print(f"MAPE: {{diagnostics.get('mape', 'N/A')}}")
    print(f"Plots saved: {{viz_results.get('plots_saved', [])}}")
'''
    
    def _assemble_code(self, header: str, modules: dict, main: str) -> str:
        """Assemble all modules into complete code"""
        sections = [header]
        
        # Add module separators
        section_titles = {
            'exploration': 'DATA EXPLORATION',
            'preprocessing': 'DATA PREPROCESSING',
            'modeling': 'BAYESIAN MODEL',
            'diagnostics': 'MODEL DIAGNOSTICS',
            'optimization': 'BUDGET OPTIMIZATION',
            'visualization': 'VISUALIZATION',
        }
        
        for name, code in modules.items():
            title = section_titles.get(name, name.upper())
            sections.append(f"""
# =============================================================================
# {title}
# =============================================================================

{code}
""")
        
        sections.append(main)
        
        return "\n".join(sections)

