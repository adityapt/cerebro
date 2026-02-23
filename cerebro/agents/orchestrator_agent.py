"""
 Orchestrator Agent with GOT Reasoning

Intelligently coordinates all specialized agents with validation and execution.

"""
import os
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from cerebro.spec.schema import MMMSpec
from cerebro.llm import AutoBackend
from cerebro.agents.base_agent import BaseAgent
from cerebro.agents.data_exploration_agent import DataExplorationAgent
from cerebro.agents.preprocessing_agent import PreprocessingAgent
from cerebro.agents.modeling_agent import ModelingAgent
from cerebro.agents.diagnostics_agent import DiagnosticsAgent
from cerebro.agents.optimization_agent import OptimizationAgent
from cerebro.agents.visualization_agent import VisualizationAgent
from cerebro.agents.hybrid_validator import HybridValidator
from cerebro.agents.pipeline_context import PipelineContext, AgentStatus
from cerebro.llm.rag_backend import RAGBackend

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """
    Orchestrates multiple specialized agents with GOT reasoning, validation, and execution.
    
    Architecture:
    
       OrchestratorAgent (GOT Reasoning) 
         [Reasons about pipeline strategy]
              
         -> DataExplorationAgent -> Generate -> Validate -> Execute -> Pass results
         -> PreprocessingAgent   -> Generate -> Validate -> Execute -> Pass results
         -> ModelingAgent        -> Generate -> Validate -> Execute -> Pass results
         -> DiagnosticsAgent     -> Generate -> Validate -> Execute -> Pass results
         -> OptimizationAgent    -> Generate -> Validate -> Execute -> Pass results
         -> VisualizationAgent   -> Generate -> Validate -> Execute -> Pass results
    
    Features:
    - GOT reasoning before orchestration (understands dependencies, validates strategy)
    - HybridValidator with up to 15 retries per module
    - RAG-enhanced code generation
    - Modular execution (separate files) or monolithic (one file)
    - Context passing between modules
    - Execution in foreground with error handling
    """
    
    def __init__(self, llm: AutoBackend, use_rag: bool = True, validator: Optional[HybridValidator] = None):
        super().__init__(llm, "OrchestratorAgent")
        self.use_rag = use_rag
        
        # Initialize validator
        if validator:
            self.validator = validator
        else:
            rag_backend = RAGBackend() if use_rag else None
            self.validator = HybridValidator(llm, max_retries=15, rag=rag_backend)
        
        # Initialize all specialist agents
        self.agents = {
            'exploration': DataExplorationAgent(llm, use_rag),
            'preprocessing': PreprocessingAgent(llm, use_rag),
            'modeling': ModelingAgent(llm, use_rag),
            'diagnostics': DiagnosticsAgent(llm, use_rag),
            'optimization': OptimizationAgent(llm, use_rag),
            'visualization': VisualizationAgent(llm, use_rag),
        }
    
        # Module metadata
        self.module_info = {
            'exploration': {
                'name': 'DATA EXPLORATION',
                'entry_point': 'run_exploration',
                'generator': 'generate_eda_code'
            },
            'preprocessing': {
                'name': 'PREPROCESSING',
                'entry_point': 'run_preprocessing',
                'generator': 'generate_preprocessing_code'
            },
            'modeling': {
                'name': 'MODELING',
                'entry_point': 'run_modeling',
                'generator': 'generate_model_code'
            },
            'diagnostics': {
                'name': 'DIAGNOSTICS',
                'entry_point': 'run_diagnostics',
                'generator': 'generate_diagnostics_code'
            },
            'optimization': {
                'name': 'OPTIMIZATION',
                'entry_point': 'run_optimization',
                'generator': 'generate_optimization_code'
            },
            'visualization': {
                'name': 'VISUALIZATION',
                'entry_point': 'run_visualization',
                'generator': 'generate_visualization_code'
            }
        }
    
    def _get_reasoning_prompt(self, spec: MMMSpec, data_path: str) -> str:
        """Generate GOT reasoning prompt for orchestration strategy"""
        return f"""
You are orchestrating a complete MMM pipeline. Reason carefully about the strategy.

SPEC:
- Model: {spec.name}
- Channels: {len(spec.channels)} marketing channels
- Controls: {len(spec.controls) if spec.controls else 0} control variables
- Outcome: {spec.outcome}
- Backend: {spec.inference.backend}
- Data: {data_path}

CRITICAL REASONING QUESTIONS (ANSWER ALL):

1) MODULE DEPENDENCIES: What is the correct execution order?
   - Which modules depend on outputs from previous modules?
   - Example: Modeling needs preprocessed data, Visualization needs model results

2) DATA FLOW: How does data pass between modules?
   - Exploration �� insights (no data modification)
   - Preprocessing �� preprocessed_data_path (CSV file)
   - Modeling �� model_results_path (JSON manifest with predictions, params, metadata)
   - Diagnostics �� diagnostics_results (dict)
   - Optimization �� optimal_allocation (dict)
   - Visualization �� plot files (PNG files)

3) VALIDATION STRATEGY: What can go wrong in each module?
   - Exploration: Date parsing, panel data handling, plot generation
   - Preprocessing: NaN/inf values, overly aggressive transformations, missing plots
   - Modeling: MCMC convergence, parameter mapping, manifest completeness
   - Diagnostics: Missing model outputs
   - Optimization: Invalid budget constraints
   - Visualization: KeyError (parameter mapping), data=None, missing plots

4) ERROR RECOVERY: What retry strategies are needed?
   - HybridValidator with up to 15 retries per module
   - Each module validated independently before execution
   - If module fails, should pipeline stop or continue?

5) EXECUTION MODE: Modular (separate files) or Monolithic (one file)?
   - Modular: Generate + Validate + Execute each module sequentially
   - Monolithic: Generate all, assemble into one file, return for user execution

OUTPUT JSON: {{"module_order": ["exploration", "preprocessing", "modeling", "diagnostics", "optimization", "visualization"], "data_flow": {{"exploration": "insights", "preprocessing": "data_path", "modeling": "results_json", "diagnostics": "diagnostics_dict", "optimization": "allocation_dict", "visualization": "plot_files"}}, "validation_needed": true, "execution_mode": "modular", "error_strategy": "stop_on_critical_failure"}}

JSON only. THINK: Modular mode allows validation + execution per module.
"""
    
    def run_pipeline(
        self,
        spec: MMMSpec,
        data_path: str,
        mode: str = "modular",  # "modular" or "monolithic"
        validate: bool = True,
        execute: bool = True,
        save_modules: bool = True,
        start_from: Optional[str] = None  # e.g. "diagnostics" to skip exploration/preprocessing/modeling
    ) -> PipelineContext:
        """
        Run complete MMM pipeline with GOT reasoning, validation, and execution.
        
        Args:
            spec: MMMSpec object defining the model
            data_path: Path to input data CSV
            mode: "modular" (separate files + validation + execution) or "monolithic" (one file)
            validate: Whether to validate each module
            execute: Whether to execute each module
            save_modules: Whether to save module files
            
        Returns:
            context: PipelineContext with full execution history and results
        """
        logger.info(" ORCHESTRATOR: INTELLIGENT PIPELINE ORCHESTRATION")
        logger.info(f"\nModel: {spec.name}")
        logger.info(f"Channels: {len(spec.channels)}")
        logger.info(f"Backend: {spec.inference.backend}")
        logger.info(f"Mode: {mode}")
        logger.info(f"Validate: {validate}")
        logger.info(f"Execute: {execute}")
        
        # STEP 1: GOT REASONING
        print("\n" + "=" * 80)
        print("STEP 1: GOT REASONING - Analyzing Pipeline Strategy")
        print("=" * 80)
        
        reasoning = self._reason_about_task(spec, {'data_path': data_path})
        
        if reasoning:
            logger.info(f"\n[GOT] Module order: {reasoning.get('module_order', [])}")
            logger.info(f"[GOT] Execution mode: {reasoning.get('execution_mode', 'modular')}")
            logger.info(f"[GOT] Validation needed: {reasoning.get('validation_needed', True)}")
        
        # STEP 2: SEQUENTIAL GENERATION -> VALIDATION -> EXECUTION
        if mode == "modular":
            return self._run_modular_pipeline(spec, data_path, validate, execute, save_modules, start_from=start_from)
        else:
            # For monolithic mode, wrap in context
            result = self._run_monolithic_pipeline(spec, data_path)
            context = PipelineContext(
                data_path=data_path,
                status=AgentStatus.SUCCESS,
                results={"monolithic": result},
                next_agent=None
            )
            return context
    
    def _run_modular_pipeline(
        self,
        spec: MMMSpec,
        data_path: str,
        validate: bool,
        execute: bool,
        save_modules: bool,
        start_from: Optional[str] = None
    ) -> PipelineContext:
        """
        Run pipeline in modular mode: Generate -> Validate -> Execute for each module.
        Uses PipelineContext for standardized agent communication.
        start_from: if "diagnostics", skip exploration/preprocessing/modeling and run from diagnostics onward (requires module_3_results.json and data_path to preprocessed CSV).
        """
        print("\n" + "=" * 80)
        print("STEP 2: MODULAR PIPELINE - Generate + Validate + Execute Each Module")
        print("=" * 80)
        if start_from:
            print(f"  [NOTE] start_from={start_from}: skipping earlier modules.")
        if not execute:
            print("  [NOTE] Execute=OFF: code will be generated and validated only (no module execution).")
        else:
            print("  [NOTE] Execute=ON: each module will run after generation and validation.")
        
        # Initialize PipelineContext
        context = PipelineContext(
            data_path=data_path,
            next_agent="diagnostics" if start_from == "diagnostics" else "exploration",
            checkpoint_path=".checkpoints",
            metadata={
                "spec_name": spec.name,
                "n_channels": len(spec.channels),
                "backend": spec.inference.backend,
                "validate": validate,
                "execute": execute
            }
        )
        if start_from == "diagnostics":
            if os.path.isfile("module_3_results.json"):
                context.manifest_path = "module_3_results.json"
                logger.info(f" start_from=diagnostics: using manifest_path={context.manifest_path}")
                print(f"  Using manifest: {context.manifest_path}")
            else:
                logger.warning(" start_from=diagnostics but module_3_results.json not found; diagnostics may be skipped.")
                print("  >> module_3_results.json not found; ensure you have run modeling first or are in the correct directory.")
        
        logger.info(f"Pipeline started: run_id={context.run_id}")
        
        # For backwards compatibility, also track in dict format
        results = {
            'modules': {},
            'executions': {},
            'fixes_applied': {},
            'errors': {}
        }
        
        # MODULE 1: EXPLORATION (skip if start_from == "diagnostics")
        if start_from != "diagnostics":
            print("\n" + "=" * 80)
            print("MODULE 1: DATA EXPLORATION - GENERATE & VALIDATE & EXECUTE")
            print("=" * 80)
            
            context.start_agent('exploration')
            
            code, exec_result = self._generate_validate_execute_module(
                module_key='exploration',
                spec=spec,
                data_path=context.data_path,
                test_args=(context.data_path,),
                validate=validate,
                execute=execute,
                save=save_modules
            )
            
            results['modules']['exploration'] = code
            results['executions']['exploration'] = exec_result
            
            if exec_result:
                context.mark_agent_complete('exploration', exec_result, next_agent='preprocessing')
            else:
                context.mark_agent_failed('exploration', "Execution returned None")
        
        # MODULE 2: PREPROCESSING (skip if start_from == "diagnostics")
        if start_from != "diagnostics":
            print("\n" + "=" * 80)
            print("MODULE 2: PREPROCESSING - GENERATE & VALIDATE & EXECUTE")
            print("=" * 80)
            
            context.start_agent('preprocessing')
            
            code, exec_result = self._generate_validate_execute_module(
                module_key='preprocessing',
                spec=spec,
                data_path=context.data_path,
                test_args=(context.data_path,),
                validate=validate,
                execute=execute,
                save=save_modules
            )
            
            results['modules']['preprocessing'] = code
            results['executions']['preprocessing'] = exec_result
            
            # Update context so modeling receives PREPROCESSED data path (critical for MMM)
            preprocessed_path = None
            if exec_result is not None:
                if isinstance(exec_result, str):
                    if exec_result.endswith('.csv') or 'preprocessed' in exec_result.lower():
                        preprocessed_path = exec_result
                elif isinstance(exec_result, dict):
                    for key in ('output_path', 'preprocessed_path', 'path', 'data_path'):
                        v = exec_result.get(key)
                        if isinstance(v, str) and (v.endswith('.csv') or 'preprocessed' in v.lower()):
                            preprocessed_path = v
                            break
            if preprocessed_path:
                context.data_path = preprocessed_path
                logger.info(f" Using preprocessed data: {context.data_path}")
                print(f"  Using preprocessed data: {context.data_path}")
                context.mark_agent_complete('preprocessing', {"preprocessed_path": preprocessed_path}, next_agent='modeling')
            elif exec_result is not None:
                context.mark_agent_complete('preprocessing', exec_result, next_agent='modeling')
                logger.warning(" Preprocessing did not return a .csv path; modeling may receive raw data path")
                print("  >> WARNING: Preprocessing output not recognized as CSV path; modeling will use current context data path.")
            else:
                context.mark_agent_failed('preprocessing', "Execution returned None")
            
            # MODULE 3: MODELING (must receive preprocessed data path from context.data_path)
            print("\n" + "=" * 80)
            print("MODULE 3: MODELING - GENERATE & VALIDATE & EXECUTE")
            print("=" * 80)
            modeling_data_path = context.data_path
            logger.info(f" Modeling input data: {modeling_data_path}")
            print(f"  Modeling input data: {modeling_data_path}")
            
            context.start_agent('modeling')
            code, exec_result = None, None
            modeling_error = None
            try:
                code, exec_result = self._generate_validate_execute_module(
                    module_key='modeling',
                    spec=spec,
                    data_path=modeling_data_path,
                    test_args=(modeling_data_path,),
                    validate=validate,
                    execute=execute,
                    save=save_modules
                )
            except Exception as e:
                modeling_error = str(e)
                logger.exception("Modeling module failed")
                print(f"\n  *** MODELING FAILED (exception): {e}")
                if code:
                    results['modules']['modeling'] = code
                    if save_modules:
                        try:
                            with open("module_modeling.py", 'w') as f:
                                f.write(code)
                            print(f"  (Saved partial code to module_modeling.py)")
                        except Exception:
                            pass
                else:
                    results['modules']['modeling'] = ""
                results['executions']['modeling'] = None
                context.mark_agent_failed('modeling', f"Exception: {e}", skip=False)
                # If manifest was written before crash, downstream can still run
                if not context.manifest_path and os.path.isfile("module_3_results.json"):
                    context.manifest_path = "module_3_results.json"
                    print(f"  >> Using existing module_3_results.json so diagnostics/optimization/visualization can run.")
            
            if modeling_error is None:
                results['modules']['modeling'] = code or ""
                results['executions']['modeling'] = exec_result
                
                # Update context manifest_path - handle both string and dict returns
                if exec_result:
                    if isinstance(exec_result, str) and exec_result.endswith('.json'):
                        context.manifest_path = exec_result
                        logger.info(f" Model results saved: {context.manifest_path}")
                        context.mark_agent_complete('modeling', {"manifest_path": exec_result}, next_agent='diagnostics')
                    elif isinstance(exec_result, dict):
                        for key in ('manifest_path', 'manifest', 'results_path'):
                            v = exec_result.get(key)
                            if isinstance(v, str) and v.endswith('.json'):
                                context.manifest_path = v
                                break
                        if not context.manifest_path and exec_result:
                            for v in exec_result.values():
                                if isinstance(v, str) and v.endswith('.json'):
                                    context.manifest_path = v
                                    break
                        if isinstance(context.manifest_path, str) and context.manifest_path.endswith('.json'):
                            logger.info(f" Model results manifest: {context.manifest_path}")
                        context.mark_agent_complete('modeling', exec_result, next_agent='diagnostics')
                    else:
                        context.mark_agent_complete('modeling', exec_result, next_agent='diagnostics')
                else:
                    context.mark_agent_failed('modeling', "Execution returned None")
                    print(f"\n  *** MODELING execution returned None (no manifest). Diagnostics/optimization/visualization will be skipped.")
                # If exec ran but returned a dict without manifest_path, try default path
                if not context.manifest_path and os.path.isfile("module_3_results.json"):
                    context.manifest_path = "module_3_results.json"
                    print(f"  Using existing module_3_results.json as manifest.")
        
        # MODULE 4: DIAGNOSTICS
        if context.manifest_path:
            print("\n" + "=" * 80)
            print("MODULE 4: DIAGNOSTICS - GENERATE & VALIDATE & EXECUTE")
            print("=" * 80)
            
            context.start_agent('diagnostics')
            
            # Load modeling output schema for data-aware generation
            upstream_output = None
            if os.path.exists(context.manifest_path):
                try:
                    import json
                    with open(context.manifest_path) as f:
                        upstream_output = json.load(f)
                    logger.info(f"[DATA-AWARE] Loaded modeling schema: {list(upstream_output.keys())}")
                except Exception as e:
                    logger.warning(f"Could not load modeling schema: {e}")
            
            code, exec_result = self._generate_validate_execute_module(
                module_key='diagnostics',
                spec=spec,
                data_path=context.data_path,
                test_args=(context.manifest_path,),
                validate=validate,
                execute=execute,
                save=save_modules,
                upstream_output=upstream_output  # Pass modeling schema
            )
            
            results['modules']['diagnostics'] = code
            results['executions']['diagnostics'] = exec_result
            
            if exec_result:
                context.mark_agent_complete('diagnostics', exec_result, next_agent='optimization')
            else:
                context.mark_agent_failed('diagnostics', "Execution returned None", skip=True)
        else:
            logger.warning("Skipping diagnostics: no manifest_path")
            print("  >> Skipping diagnostics: no manifest_path (modeling failed or did not return a manifest).")
            context.skipped_agents.append('diagnostics')
        
        # MODULE 5: OPTIMIZATION
        if context.manifest_path:
            print("\n" + "=" * 80)
            print("MODULE 5: OPTIMIZATION - GENERATE & VALIDATE & EXECUTE")
            print("=" * 80)
            
            context.start_agent('optimization')
            
            code, exec_result = self._generate_validate_execute_module(
                module_key='optimization',
                spec=spec,
                data_path=context.data_path,
                test_args=(context.manifest_path,),
                validate=validate,
                execute=execute,
                save=save_modules
            )
            
            results['modules']['optimization'] = code
            results['executions']['optimization'] = exec_result
            
            if exec_result:
                context.mark_agent_complete('optimization', exec_result, next_agent='visualization')
            else:
                context.mark_agent_failed('optimization', "Execution returned None", skip=True)
        else:
            logger.warning("Skipping optimization: no manifest_path")
            print("  >> Skipping optimization: no manifest_path (modeling failed or did not return a manifest).")
            context.skipped_agents.append('optimization')
        
        # MODULE 6: VISUALIZATION
        if context.manifest_path:
            print("\n" + "=" * 80)
            print("MODULE 6: VISUALIZATION - GENERATE & VALIDATE & EXECUTE")
            print("=" * 80)
            
            context.start_agent('visualization')
            
            code, exec_result = self._generate_validate_execute_module(
                module_key='visualization',
                spec=spec,
                data_path=context.data_path,
                test_args=(context.manifest_path,),
                validate=validate,
                execute=execute,
                save=save_modules
            )
            
            results['modules']['visualization'] = code
            results['executions']['visualization'] = exec_result
            
            if exec_result:
                context.mark_agent_complete('visualization', exec_result, next_agent=None)  # Pipeline complete
            else:
                context.mark_agent_failed('visualization', "Execution returned None", skip=True)
        else:
            logger.warning("Skipping visualization: no manifest_path")
            print("  >> Skipping visualization: no manifest_path (modeling failed or did not return a manifest).")
            context.skipped_agents.append('visualization')
        
        # FINAL SUMMARY
        print("\n" + "=" * 80)
        print(" PIPELINE COMPLETE!")
        print("=" * 80)
        
        # Mark pipeline as complete
        context.next_agent = None
        if context.status != AgentStatus.FAILED:
            context.status = AgentStatus.SUCCESS
        
        # For backwards compatibility, also store in context.results
        context.results['_legacy_format'] = results
        
        print(f"\nModules Generated: {len(results['modules'])}")
        for name, code in results['modules'].items():
            print(f"  {name:15s}: {len(code.splitlines())} lines")
        
        print(f"\nModules Executed: {len([r for r in results['executions'].values() if r is not None])}")
        print(f"Completed: {context.completed_agents}")
        print(f"Skipped: {context.skipped_agents if context.skipped_agents else 'None'}")
        
        if save_modules:
            print(f"\nSaved Files:")
            for name in results['modules'].keys():
                print(f"  module_{name}.py")
        
        # Log performance
        perf = context.get_performance_summary()
        if perf:
            print(f"\nPerformance:")
            for agent, duration_ms in perf.items():
                print(f"  {agent:15s}: {duration_ms}ms")
        
        # Save checkpoint
        checkpoint_file = context.save_checkpoint()
        if checkpoint_file:
            logger.info(f"Checkpoint saved: {checkpoint_file}")
        
        return context
    
    def _generate_validate_execute_module(
        self,
        module_key: str,
        spec: MMMSpec,
        data_path: str,
        test_args: Tuple,
        validate: bool,
        execute: bool,
        save: bool,
        upstream_output: dict = None
    ) -> Tuple[str, Any]:
        """
        Generate, validate, and execute a single module.
        
        Args:
            upstream_output: Schema/manifest from upstream module (for data-aware generation)
        
        Returns:
            (code, execution_result)
        """
        info = self.module_info[module_key]
        agent = self.agents[module_key]
        
        # GENERATE (pass upstream_output if agent supports it)
        print(f"Generating {info['name'].lower()} code...\n")
        generator_method = getattr(agent, info['generator'])
        
        # Try to pass upstream_output if method accepts it
        import inspect
        sig = inspect.signature(generator_method)
        if 'upstream_output' in sig.parameters and upstream_output:
            code = generator_method(spec=spec, data_path=data_path, upstream_output=upstream_output)
        else:
            code = generator_method(spec=spec, data_path=data_path)
        print(f"\nGenerated: {len(code.splitlines())} lines")
        
        # VALIDATE
        if validate:
            print(f"\nValidating with execution testing (up to 15 retries)...")
            code, fixes = self.validator.validate_and_fix(
                code=code,
                spec=spec,
                module_name=module_key,
                entry_point=info['entry_point'],
                test_args=test_args
            )
            
            if fixes:
                print(f"  Applied fixes in {len(fixes)} attempts:")
                for attempt, fix_list in fixes.items():
                    print(f"    {attempt}: {fix_list}")
            print(f"  Validation PASSED!")
        
        # SAVE
        if save:
            filename = f"module_{module_key}.py"
            with open(filename, 'w') as f:
                f.write(code)
            print(f"Saved: {filename}")
        if not execute and module_key == 'modeling':
            print("  (Execute is OFF; modeling code was not run. Run with execute=True to execute.)")
        
        # EXECUTE
        exec_result = None
        if execute:
            print(f"\nExecuting {info['name'].lower()} with FULL configuration...")
            if module_key == 'modeling' and hasattr(spec, 'inference'):
                nw = getattr(spec.inference, 'num_warmup', 500)
                ns = getattr(spec.inference, 'num_samples', 500)
                nc = getattr(spec.inference, 'num_chains', 1)
                print(f"  Full MCMC: {nw} warmup, {ns} samples, {nc} chain(s)")
            # Force line-buffered stdout/stderr so MCMC progress shows when piped (e.g. tee)
            if module_key == 'modeling':
                import sys
                try:
                    if hasattr(sys.stdout, 'reconfigure'):
                        sys.stdout.reconfigure(line_buffering=True)
                    if hasattr(sys.stderr, 'reconfigure'):
                        sys.stderr.reconfigure(line_buffering=True)
                    sys.stdout.flush()
                    sys.stderr.flush()
                except Exception:
                    pass
            try:
                exec_globals = {}
                exec(code, exec_globals)
                
                entry_point = info['entry_point']
                if entry_point in exec_globals:
                    exec_result = exec_globals[entry_point](*test_args)
                    print(f"{info['name']} completed!")
                    print(f"  Output: {exec_result}")
                    if module_key == 'modeling' and exec_result is None:
                        print(f"  >> WARNING: {entry_point}() returned None. Downstream modules need a dict with 'manifest_path' (e.g. 'module_3_results.json'). Check that the generated run_modeling returns the results dict.")
                else:
                    print(f"  ERROR: No {entry_point} function found in generated code. Check module_{module_key}.py defines def {entry_point}(...).")
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                traceback.print_exc()
                if module_key == 'modeling':
                    print(f"  >> Modeling execution crashed. Fix the error above or check module_modeling.py (e.g. wrong data path, missing deps, or model/code bug).")
        
        return code, exec_result
    
    def _run_monolithic_pipeline(self, spec: MMMSpec, data_path: str) -> Dict[str, Any]:
        """
        Run pipeline in monolithic mode: Generate all modules, assemble into one file.
        """
        print("\n" + "=" * 80)
        print("STEP 2: MONOLITHIC PIPELINE - Generate All Modules")
        print("=" * 80)
        
        # Generate header
        header = self._generate_header(spec)
        
        # Generate all modules
        modules = {}
        
        print("\nGenerating all modules...")
        for key, agent in self.agents.items():
            info = self.module_info[key]
            print(f"  {info['name']}...", end=' ')
            generator_method = getattr(agent, info['generator'])
            code = generator_method(spec=spec, data_path=data_path)
            modules[key] = code
            print(f" {len(code.splitlines())} lines")
        
        # Generate main execution
        main_execution = self._generate_main(spec, data_path)
        
        # Assemble complete code
        complete_code = self._assemble_code(header, modules, main_execution)
        
        # Save
        output_path = "complete_mmm_pipeline.py"
        with open(output_path, 'w') as f:
            f.write(complete_code)
        
        print(f"\n Complete pipeline saved: {output_path}")
        print(f"  Total: {len(complete_code.splitlines())} lines")
        
        return {
            'complete_code': complete_code,
            'modules': modules,
            'output_path': output_path
        }
    
    def _generate_header(self, spec: MMMSpec) -> str:
        """Generate header with imports"""
        return f'''"""
 AUTO-GENERATED MMM PIPELINE

Model: {spec.name}
Channels: {len(spec.channels)}
Backend: {spec.inference.backend}
Generated by: Cerebro OrchestratorAgent with GOT Reasoning

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
jax.config.update('jax_platform_name', 'cpu')

print("=" * 80)
print(" MMM PIPELINE")
print("=" * 80)
print(f"Model: {spec.name}")
print(f"Channels: {len(spec.channels)}")
print(f"Backend: {spec.inference.backend}")
print("=" * 80)

'''
    
    def _generate_main(self, spec: MMMSpec, data_path: str) -> str:
        """Generate main execution code"""
        channel_names = [ch.name for ch in spec.channels]
        control_names = spec.controls if spec.controls else []
        
        return f'''

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # Load data
    print("\\nLoading data from: {data_path}")
    data_df = pd.read_csv('{data_path}')
    print(f" Loaded {{len(data_df)}} rows, {{len(data_df.columns)}} columns")
    
    # Define columns
    outcome_col = '{spec.outcome}'
    channel_cols = {channel_names}
    control_cols = {control_names}
    date_col = '{spec.date_column if hasattr(spec, "date_column") else None}'
    
    # Step 1: Explore data
    eda_result = run_exploration('{data_path}')
    
    # Step 2: Preprocess
    preprocessed_path = run_preprocessing('{data_path}')
    
    # Step 3: Model
    model_results_path = run_modeling(preprocessed_path)
    
    # Step 4: Diagnostics
    diagnostics_result = run_diagnostics(model_results_path)
    
    # Step 5: Optimization
    optimization_result = run_optimization(model_results_path)
    
    # Step 6: Visualization
    viz_result = run_visualization(model_results_path)
    
    print("\\n" + "=" * 80)
    print(" COMPLETE MMM PIPELINE EXECUTED SUCCESSFULLY!")
    print("=" * 80)
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
