"""
Demo: Auto-detecting optimal backend for your environment

This script demonstrates Cerebro's intelligent backend selection:
- Mac M1/M2/M3 → MLX (5x faster)
- NVIDIA GPU   → vLLM (2-5x faster)
- CPU/Other    → Ollama (universal)
"""

import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import Cerebro
from cerebro.llm import AutoBackend, get_optimal_backend
from cerebro.agents import MMMOrchestrator


def demo_auto_detection():
    """Demonstrate automatic backend detection"""
    print("=" * 80)
    print("CEREBRO AUTO-BACKEND DETECTION")
    print("=" * 80)
    print()
    
    # Auto-detect optimal backend
    print("🔍 Detecting your environment...")
    llm = AutoBackend(model="qwen2.5-coder:32b")
    
    stats = llm.get_stats()
    print()
    print("✅ Selected Backend:")
    print(f"   Backend: {stats['backend']}")
    print(f"   Device: {stats['device']}")
    print(f"   Optimized for: {stats['optimized_for']}")
    print(f"   Expected speed: {stats['expected_speed']}")
    print()


def demo_manual_selection():
    """Demonstrate manual backend selection"""
    print("=" * 80)
    print("MANUAL BACKEND SELECTION")
    print("=" * 80)
    print()
    
    backends = {
        'mlx': 'Apple Silicon (M1/M2/M3)',
        'vllm': 'NVIDIA GPU (Databricks, AWS, GCP)',
        'ollama': 'Universal (CPU, any platform)'
    }
    
    for backend_name, description in backends.items():
        print(f"🔧 {backend_name.upper()}: {description}")
        try:
            llm = AutoBackend(
                model="qwen2.5-coder:32b",
                force_backend=backend_name
            )
            stats = llm.get_stats()
            print(f"   ✅ Available - {stats['expected_speed']}")
        except Exception as e:
            print(f"   ❌ Not available: {str(e)[:50]}...")
        print()


def demo_mmm_pipeline():
    """Run MMM pipeline with auto-detected backend"""
    print("=" * 80)
    print("MMM PIPELINE WITH AUTO-BACKEND")
    print("=" * 80)
    print()
    
    # Check if example data exists
    data_path = Path("/Users/adityapu/Documents/GitHub/deepcausalmmm/examples/sample_data.csv")
    
    if not data_path.exists():
        print("❌ Example data not found. Skipping pipeline demo.")
        print(f"   Expected: {data_path}")
        return
    
    # Load data
    print("📊 Loading example data...")
    data = pd.read_csv(data_path)
    print(f"   Shape: {data.shape}")
    print(f"   Columns: {list(data.columns)}")
    print()
    
    # Initialize auto-backend
    print("🚀 Initializing MMM pipeline with auto-backend...")
    llm = AutoBackend(model="qwen2.5-coder:32b")
    
    stats = llm.get_stats()
    print(f"   Using: {stats['backend']} ({stats['expected_speed']})")
    print()
    
    # Run MMM pipeline
    orchestrator = MMMOrchestrator(
        llm=llm,
        use_code_judge=True,
        use_results_judge=True
    )
    
    print("🔄 Running MMM pipeline...")
    print("   This may take 5-45 minutes depending on your hardware:")
    print(f"   - MLX (Mac):     ~5-10 minutes")
    print(f"   - vLLM (GPU):    ~2-5 minutes")
    print(f"   - Ollama (CPU):  ~30-45 minutes")
    print()
    
    try:
        results = orchestrator.run_pipeline(
            data=data,
            target_column="target_sales",
            verbose=True
        )
        
        print()
        print("=" * 80)
        print("✅ MMM PIPELINE COMPLETE")
        print("=" * 80)
        print()
        
        # Display results
        if 'model' in results:
            print("📈 MODEL RESULTS:")
            model_results = results['model']
            
            if 'coefficients' in model_results:
                print("\n   Channel Coefficients:")
                for channel, coef in model_results['coefficients'].items():
                    print(f"      {channel}: {coef:.4f}")
            
            if 'roi' in model_results:
                print("\n   ROI by Channel:")
                for channel, roi in model_results['roi'].items():
                    print(f"      {channel}: {roi:.2f}x")
            
            if 'r_squared' in model_results:
                print(f"\n   R²: {model_results['r_squared']:.4f}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all demos"""
    # Demo 1: Auto-detection
    demo_auto_detection()
    
    # Demo 2: Manual selection
    demo_manual_selection()
    
    # Demo 3: Full MMM pipeline (optional)
    run_pipeline = input("Run full MMM pipeline demo? (y/n): ").lower().strip()
    if run_pipeline == 'y':
        demo_mmm_pipeline()
    else:
        print("Skipping pipeline demo.")
    
    print()
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. On Mac: MLX will be 5x faster than Ollama")
    print("2. On Databricks: Use vLLM for 2-5x faster inference")
    print("3. On CPU: Ollama provides universal compatibility")
    print()
    print("To force a specific backend:")
    print("   llm = AutoBackend(model='qwen2.5-coder:32b', force_backend='mlx')")


if __name__ == "__main__":
    main()


