#!/usr/bin/env python
"""
Cerebro CLI - Command-line interface for autonomous MMM agent
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Cerebro: Autonomous Marketing Mix Modeling Agent"
    )
    parser.add_argument(
        "data_path",
        help="Path to CSV file with MMM data"
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:32b",
        help="Ollama model to use (default: qwen2.5-coder:32b)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout for LLM generation in seconds (default: 600)"
    )
    
    args = parser.parse_args()
    
    # Import here to avoid slow imports on --help
    from cerebro.llm.qwen_rag_backend import QwenRAGBackend
    import pandas as pd
    import time
    
    print("=" * 80)
    print("CEREBRO - AUTONOMOUS MMM AGENT")
    print("=" * 80)
    print(f"\n📂 Data: {args.data_path}")
    print(f"🤖 Model: {args.model}")
    print()
    
    # Initialize agent (RAG loads instantly from disk!)
    print("⏳ Loading agent...")
    start = time.time()
    llm = QwenRAGBackend(model=args.model, timeout=args.timeout)
    print(f"✅ Agent ready in {time.time() - start:.1f}s\n")
    
    # Load data
    print(f"📊 Loading data...")
    data = pd.read_csv(args.data_path)
    print(f"   Shape: {data.shape}")
    print(f"   Columns: {list(data.columns[:5])}{'...' if len(data.columns) > 5 else ''}\n")
    
    # Build prompt
    prompt = f"""
You are exploring a marketing dataset with {len(data)} rows and {len(data.columns)} columns.

TASK: Explore this data autonomously and build a production-grade MMM model.

1. Identify target, media channels, controls, and date column
2. Analyze autocorrelation and decide on adstock transformations
3. Check multicollinearity and choose modeling approach
4. Build the model (Bayesian if appropriate, otherwise frequentist with positive constraints)
5. Calculate ROI for each channel
6. Print model summary and ROI

Generate complete Python code. The data is already loaded as `data` DataFrame.
"""
    
    print("🔄 Agent is exploring and generating code...\n")
    print("-" * 80)
    
    # Generate code
    code = llm.generate(prompt, use_rag=True)
    
    print("-" * 80)
    print(f"\n✅ Code generated!\n")
    
    # Execute code
    print("="*80)
    print("EXECUTING GENERATED CODE")
    print("="*80)
    print()
    
    namespace = {'data': data}
    try:
        exec(code, namespace)
        print("\n✅ Model built successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save code for debugging
        debug_file = Path("/tmp/cerebro_debug.py")
        with open(debug_file, 'w') as f:
            f.write(code)
        print(f"\n💾 Code saved to: {debug_file}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

