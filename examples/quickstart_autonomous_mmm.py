#!/usr/bin/env python3
"""
🚀 Quickstart: Autonomous MMM Agent

This example shows how to use Cerebro to automatically:
1. Analyze your MMM data
2. Generate a structured YAML spec
3. Generate production-grade Python code
4. Create a complete MMM pipeline

Usage:
    python examples/quickstart_autonomous_mmm.py path/to/data.csv

Requirements:
    - Data CSV with: date column, outcome column, media channels, controls
    - Ollama running locally (or set OPENAI_API_KEY for API models)
"""

import sys
from pathlib import Path

# Add cerebro to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cerebro.cli import main as cerebro_main

def quickstart_example():
    """
    Quickstart example showing autonomous MMM generation
    """
    print("=" * 80)
    print("🚀 CEREBRO QUICKSTART: AUTONOMOUS MMM")
    print("=" * 80)
    print()
    
    # Example data path (replace with your data)
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        print("Usage: python quickstart_autonomous_mmm.py path/to/data.csv")
        print()
        print("Example:")
        print("  python examples/quickstart_autonomous_mmm.py ~/data/mmm_data.csv")
        print()
        print("Expected CSV columns:")
        print("  - Date column (e.g., 'date', 'week', 'weekid')")
        print("  - Outcome column (e.g., 'sales', 'revenue', 'conversions')")
        print("  - Channel columns (e.g., 'tv_spend', 'digital_impressions')")
        print("  - Control columns (optional, e.g., 'price', 'promotion')")
        return
    
    print(f"📁 Data: {data_path}")
    print()
    print("🤖 The agent will:")
    print("  1. Profile your data (columns, types, distributions)")
    print("  2. Generate a structured MMM spec (YAML)")
    print("  3. Generate production Python code (~1000+ lines)")
    print("  4. Save outputs to ./generated/")
    print()
    print("⏱️  This may take 2-5 minutes with local LLM...")
    print("=" * 80)
    print()
    
    # Run Cerebro autonomous pipeline
    sys.argv = [
        'cerebro',
        'auto',
        data_path,
        '--output', './generated/mmm_pipeline.py',
        '--save-spec'
    ]
    
    cerebro_main()
    
    print()
    print("=" * 80)
    print("✅ GENERATION COMPLETE!")
    print("=" * 80)
    print()
    print("📂 Generated files:")
    print("  - ./generated/mmm_pipeline.py   (Full Python pipeline)")
    print("  - ./generated/mmm_spec.yaml     (Model specification)")
    print()
    print("🚀 Next steps:")
    print("  1. Review the generated spec: cat ./generated/mmm_spec.yaml")
    print("  2. Run the pipeline: python ./generated/mmm_pipeline.py")
    print("  3. Customize as needed!")
    print()

if __name__ == '__main__':
    quickstart_example()

