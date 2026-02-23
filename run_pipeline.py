"""
Cerebro MMM Pipeline - Intelligent Orchestration

Uses OrchestratorAgent with GOT reasoning for intelligent pipeline execution.
"""

import sys
import argparse
import logging
import yaml
from pathlib import Path

# Add cerebro to path
sys.path.insert(0, str(Path(__file__).parent))

from cerebro.spec.schema import MMMSpec
from cerebro.llm.api_backend import ApiBackend
from cerebro.agents.spec_writer_agent import AutonomousSpecWriter
from cerebro.agents.orchestrator_agent import OrchestratorAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cerebro Agentic MMM Pipeline with Intelligent Orchestration')
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenAI API key (can also use OPENAI_API_KEY env var)')
    parser.add_argument('--base-url', type=str, default=None,
                        help='API base URL (can also use OPENAI_BASE_URL env var)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (default: gpt-4o)')
    parser.add_argument('--data-path', type=str, default='examples/MMM Data.csv',
                        help='Path to input data CSV file')
    parser.add_argument('--spec-path', type=str, default='spec_auto.yaml',
                        help='Path to save generated spec file')
    parser.add_argument('--mode', type=str, default='modular', choices=['modular', 'monolithic'],
                        help='Pipeline mode: modular (separate files + validation + execution) or monolithic (one file)')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip validation (not recommended)')
    parser.add_argument('--no-save', action='store_true',
                        help='Skip saving module files')
    parser.add_argument('--log-file', type=str, default='pipeline_run.log',
                        help='Path to log file (stdout + all logger output). Default: pipeline_run.log')
    parser.add_argument('--start-from', type=str, default=None, choices=['diagnostics'],
                        help='Skip earlier modules and run from this one (e.g. diagnostics). Requires module_3_results.json and preprocessed data; use --data-path for preprocessed CSV.')
    return parser.parse_args()


class Tee:
    """Write to both a file and the original stream."""
    def __init__(self, file_handle, stream):
        self.file = file_handle
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()
    def flush(self):
        self.stream.flush()
        self.file.flush()


if __name__ == '__main__':
    args = parse_args()
    
    # Log to file: resolve path relative to project root (run_pipeline.py's parent)
    project_root = Path(__file__).resolve().parent
    log_path = project_root / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file_handle = open(log_path, 'w', encoding='utf-8')
    # Send logging to both console and log file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(log_file_handle),
            logging.StreamHandler(sys.__stdout__),
        ],
        force=True,
    )
    # Tee stdout/stderr so print() and raw writes also go to the log file
    sys.stdout = Tee(log_file_handle, sys.__stdout__)
    sys.stderr = Tee(log_file_handle, sys.__stderr__)
    print(f"Logging to: {log_path.resolve()}")
    print()
    
    print("=" * 80)
    print("CEREBRO INTELLIGENT MMM PIPELINE")
    print("=" * 80)
    print("\nFeatures:")
    print("  - GOT Reasoning for intelligent orchestration")
    print("  - HybridValidator with auto-fix (up to 15 retries)")
    print("  - RAG-enhanced code generation")
    print("  - Real-time streaming output")
    print("  - Modular or monolithic generation")
    print("  - Execute generated code: YES (all modules run after generation)")
    print()

    # ============================================================================
    # STEP 1: LLM INITIALIZATION
    # ============================================================================
    print("=" * 80)
    print("STEP 1: LLM INITIALIZATION")
    print("=" * 80)
    print("Initializing GPT-4o backend...")
    
    llm = ApiBackend(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model
    )
    print("GPT-4o ready!")

    # ============================================================================
    # STEP 2: SPEC (generate or load when --start-from)
    # ============================================================================
    if args.start_from:
        print("\n" + "=" * 80)
        print("STEP 2: LOADING EXISTING SPEC (--start-from)")
        print("=" * 80)
        spec_path = Path(args.spec_path)
        if not spec_path.exists():
            print(f"  ERROR: Spec file not found: {spec_path}")
            sys.exit(1)
        with open(spec_path) as f:
            spec_dict = yaml.safe_load(f) or {}
        spec = MMMSpec(**spec_dict)
        print(f"  Loaded spec from {spec_path}: {spec.name}, {len(spec.channels)} channels")
        data_path = args.data_path
        if args.start_from == 'diagnostics' and data_path == 'examples/MMM Data.csv':
            # Default to preprocessed CSV when re-running from diagnostics
            alt = Path('examples/MMM Data_preprocessed.csv')
            if alt.exists():
                data_path = str(alt)
                print(f"  Using preprocessed data: {data_path}")
    else:
        print("\n" + "=" * 80)
        print("STEP 2: AGENTIC SPEC GENERATION")
        print("=" * 80)
        print(f"Analyzing data: {args.data_path}")
        print("Agent will infer: channels, controls, date column, outcome, priors...")
        print()

        spec_agent = AutonomousSpecWriter(llm)
        print("Generating spec with AI agent (streaming)...\n")

        spec = spec_agent.analyze_and_generate_spec(
            data_path=args.data_path,
            target_column=None  # Auto-detect
        )

        # Save the spec
        spec_agent.save_spec(spec, args.spec_path)

        print(f"\nSpec generated by agent!")
        print(f"  Date column: {spec.date_column}")
        print(f"  Outcome: {spec.outcome}")
        print(f"  Channels: {len(spec.channels)}")
        print(f"  Controls: {len(spec.controls) if spec.controls else 0}")
        print(f"  Saved to: {args.spec_path}")
        data_path = args.data_path

    # ============================================================================
    # STEP 3: INTELLIGENT ORCHESTRATION WITH GOT REASONING
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 3: INTELLIGENT ORCHESTRATION")
    print("=" * 80)
    print("Initializing OrchestratorAgent with GOT reasoning...")
    
    orchestrator = OrchestratorAgent(llm, use_rag=True)
    print("OrchestratorAgent ready!")
    print(f"  Agents: {len(orchestrator.agents)}")
    print(f"  Validator: HybridValidator (4 layers, max_retries=15, RAG-enabled)")
    print(f"  Mode: {args.mode}")
    print(f"  Execute: ON (all generated code runs after generation)")
    
    # Run the pipeline
    context = orchestrator.run_pipeline(
        spec=spec,
        data_path=data_path,
        mode=args.mode,
        validate=not args.no_validate,
        execute=True,
        save_modules=not args.no_save,
        start_from=args.start_from
    )
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("CEREBRO PIPELINE COMPLETE!")
    print("=" * 80)
    
    if args.mode == 'modular':
        # Get legacy format for backwards compatibility
        results = context.results.get('_legacy_format', {})
        
        print("\nMODULAR MODE RESULTS:")
        print(f"  Run ID: {context.run_id}")
        print(f"  Status: {context.status}")
        print(f"  Modules Completed: {len(context.completed_agents)}")
        print(f"  Modules Skipped: {len(context.skipped_agents)}")
        
        if context.completed_agents:
            print(f"\n  Completed: {', '.join(context.completed_agents)}")
        if context.skipped_agents:
            print(f"  Skipped: {', '.join(context.skipped_agents)}")
        
        if not args.no_save and results.get('modules'):
            print(f"\nGenerated Files:")
            for name in results['modules'].keys():
                print(f"    module_{name}.py")
        
        perf = context.get_performance_summary()
        if perf:
            print(f"\nPerformance:")
            for agent, duration_ms in perf.items():
                print(f"    {agent:15s}: {duration_ms}ms")
    else:
        # Monolithic mode
        print("\nMONOLITHIC MODE RESULTS:")
        monolithic_result = context.results.get('monolithic', {})
        print(f"  Output file: {monolithic_result.get('output_path', 'N/A')}")
        if 'complete_code' in monolithic_result:
            print(f"  Total lines: {len(monolithic_result['complete_code'].splitlines())}")
            print(f"\nRun with: python {monolithic_result.get('output_path', '')}")
    
    print("\n" + "=" * 80)
    print("SYSTEM STATUS")
    print("=" * 80)
    print("  [OK] Spec generated AGENTICALLY (not hardcoded!)")
    print("  [OK] GOT reasoning applied for intelligent orchestration")
    print("  [OK] All modules generated with streaming")
    print("  [OK] HybridValidator with execution testing")
    print("  [OK] RAG-enhanced code generation")
    print("  [OK] Temperature=0.0 (deterministic)")
    print("\nEverything generated by AI agents with reasoning!")
    print("=" * 80)
