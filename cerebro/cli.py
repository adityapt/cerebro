#!/usr/bin/env python
"""
Cerebro CLI - Command-line interface for autonomous product and marketing analytics

Commands:
    auto        - Autonomous analysis generation from data (spec + code)
    generate    - Generate code from existing spec
    validate    - Validate generated code with execution feedback
"""

import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_auto(args):
    """
    Autonomous Analytics: Data → Spec → Code → Validation
    """
    from cerebro.llm import AutoBackend
    from cerebro.agents.spec_writer_agent import AutonomousSpecWriter
    from cerebro.agents.orchestrator_agent import OrchestratorAgent
    from cerebro.agents.execution_validator import ExecutionValidator
    import yaml
    
    print("CEREBRO: AUTONOMOUS ANALYTICS GENERATION")
    print(f"\nData: {args.data_path}")
    print(f"LLM: {args.llm}")
    print(f"Output: {args.output}")
    print()
    
    # Initialize LLM
    print("Initializing LLM backend...")
    llm = AutoBackend.create(args.llm)
    print("LLM ready.\n")
    
    # Step 1: Generate Spec
    print("STEP 1: GENERATING ANALYTICAL SPECIFICATION")
    spec_writer = AutonomousSpecWriter(llm)
    spec = spec_writer.generate_spec_from_data(args.data_path)
    
    if args.save_spec:
        spec_path = args.output.replace('.py', '_spec.yaml')
        with open(spec_path, 'w') as f:
            yaml.dump(spec.model_dump(), f, default_flow_style=False)
        print(f"\n✓ Spec saved to: {spec_path}")
    
    # Step 2: Generate Code
    print("STEP 2: GENERATING PRODUCTION CODE")
    orchestrator = OrchestratorAgent(llm, use_rag=True)
    code = orchestrator.generate_complete_pipeline(
        spec,
        output_path=args.output,
        data_path=args.data_path
    )
    
    # Step 3: Validate with Execution Feedback
    if args.validate:
        print("STEP 3: EXECUTION VALIDATION")
        validator = ExecutionValidator(max_retries=3)
        code, success = validator.validate_and_fix(
            code,
            args.data_path,
            orchestrator.agents['modeling'],  # Use modeling agent for fixes
            context="Complete MMM pipeline"
        )
        
        if success:
            print("\n✓ Code validated successfully!")
        else:
            print("\n⚠ Code has errors but max retries reached.")
            print("  Saving code for manual review.")
    
    # Save final code
    with open(args.output, 'w') as f:
        f.write(code)
    
    print("GENERATION COMPLETE")
    print(f"\n✓ Generated code: {args.output}")
    print(f"  Lines: {len(code.splitlines())}")
    print(f"\nNext steps:")
    print(f"  1. Review the code: cat {args.output}")
    print(f"  2. Run it: python {args.output}")
    print()


def cmd_generate(args):
    """
    Generate code from existing spec
    """
    from cerebro.llm import AutoBackend
    from cerebro.agents.orchestrator_agent import OrchestratorAgent
    from cerebro.spec.schema import MMMSpec
    from cerebro.agents.execution_validator import ExecutionValidator
    import yaml
    
    print("CEREBRO: GENERATE FROM SPEC")
    print(f"\nSpec: {args.spec_path}")
    print(f"Output: {args.output}")
    print()
    
    # Load spec
    with open(args.spec_path) as f:
        spec_dict = yaml.safe_load(f)
    spec = MMMSpec(**spec_dict)
    
    # Initialize LLM
    print("Initializing LLM backend...")
    llm = AutoBackend.create(args.llm)
    print("LLM ready.\n")
    
    # Generate code
    print("GENERATING CODE FROM SPEC")
    orchestrator = OrchestratorAgent(llm, use_rag=True)
    code = orchestrator.generate_complete_pipeline(
        spec,
        output_path=args.output,
        data_path=args.data_path if args.data_path else None
    )
    
    # Validate if requested and data available
    if args.validate and args.data_path:
        print("EXECUTION VALIDATION")
        validator = ExecutionValidator(max_retries=3)
        code, success = validator.validate_and_fix(
            code,
            args.data_path,
            orchestrator.agents['modeling'],
            context=f"MMM pipeline for {spec.name}"
        )
        
        if success:
            print("\n✓ Code validated successfully!")
        else:
            print("\n⚠ Code has errors but max retries reached.")
    
    # Save
    with open(args.output, 'w') as f:
        f.write(code)
    
    print("GENERATION COMPLETE")
    print(f"\n✓ Generated: {args.output}")
    print(f"  Lines: {len(code.splitlines())}")
    print()


def cmd_validate(args):
    """
    Validate existing generated code
    """
    from cerebro.llm import AutoBackend
    from cerebro.agents.modeling_agent import ModelingAgent
    from cerebro.agents.execution_validator import ExecutionValidator
    
    print("CEREBRO: CODE VALIDATION")
    print(f"\nCode: {args.code_path}")
    print(f"Data: {args.data_path}")
    print()
    
    # Load code
    with open(args.code_path) as f:
        code = f.read()
    
    # Initialize LLM for fixes
    llm = AutoBackend.create(args.llm)
    agent = ModelingAgent(llm, use_rag=True)
    
    # Validate
    validator = ExecutionValidator(max_retries=3)
    fixed_code, success = validator.validate_and_fix(
        code,
        args.data_path,
        agent,
        context="MMM pipeline validation"
    )
    
    # Save if fixed
    if args.output:
        with open(args.output, 'w') as f:
            f.write(fixed_code)
        print(f"\n✓ Fixed code saved to: {args.output}")
    
    print("VALIDATION COMPLETE")
    print(f"\nStatus: {'✓ SUCCESS' if success else '✗ FAILED'}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Cerebro: Autonomous Product and Marketing Data Science",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Autonomous analytics from data
  cerebro auto data.csv --output pipeline.py --validate
  
  # Generate from existing spec
  cerebro generate spec.yaml --output pipeline.py --data data.csv
  
  # Validate existing code
  cerebro validate generated.py --data data.csv
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Auto command
    auto_parser = subparsers.add_parser(
        'auto',
        help='Autonomous analytics generation from data'
    )
    auto_parser.add_argument(
        'data_path',
        help='Path to CSV data file'
    )
    auto_parser.add_argument(
        '--output', '-o',
        default='generated_mmm_pipeline.py',
        help='Output path for generated code (default: generated_mmm_pipeline.py)'
    )
    auto_parser.add_argument(
        '--llm',
        default='ollama:qwen2.5:7b',
        help='LLM backend (default: ollama:qwen2.5:7b)'
    )
    auto_parser.add_argument(
        '--save-spec',
        action='store_true',
        help='Save generated spec to YAML'
    )
    auto_parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate generated code with execution feedback'
    )
    
    # Generate command
    gen_parser = subparsers.add_parser(
        'generate',
        help='Generate code from existing spec'
    )
    gen_parser.add_argument(
        'spec_path',
        help='Path to YAML spec file'
    )
    gen_parser.add_argument(
        '--output', '-o',
        default='generated_mmm_pipeline.py',
        help='Output path for generated code'
    )
    gen_parser.add_argument(
        '--data-path',
        help='Path to data (optional, needed for validation)'
    )
    gen_parser.add_argument(
        '--llm',
        default='ollama:qwen2.5:7b',
        help='LLM backend'
    )
    gen_parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate generated code (requires --data-path)'
    )
    
    # Validate command
    val_parser = subparsers.add_parser(
        'validate',
        help='Validate existing generated code'
    )
    val_parser.add_argument(
        'code_path',
        help='Path to Python code to validate'
    )
    val_parser.add_argument(
        'data_path',
        help='Path to CSV data file for testing'
    )
    val_parser.add_argument(
        '--output', '-o',
        help='Output path for fixed code (optional)'
    )
    val_parser.add_argument(
        '--llm',
        default='ollama:qwen2.5:7b',
        help='LLM backend for fixes'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to command
    if args.command == 'auto':
        cmd_auto(args)
    elif args.command == 'generate':
        cmd_generate(args)
    elif args.command == 'validate':
        cmd_validate(args)


if __name__ == "__main__":
    main()
