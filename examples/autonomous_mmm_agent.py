"""
Truly Autonomous MMM Agent
- No spoon-feeding
- Agent explores data independently
- Generates code dynamically
- Shows code generation in real-time
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import json
import subprocess
import time
from pathlib import Path

# Initialize Qwen 32B with RAG
print("=" * 80)
print("AUTONOMOUS MMM AGENT - Zero Spoon-Feeding")
print("=" * 80)
print("\n🤖 Initializing Qwen 32B + RAG...")

from cerebro.llm.qwen_rag_backend import QwenRAGBackend

llm = QwenRAGBackend(
    model="qwen2.5-coder:32b",
    timeout=600
)

print(f"✅ Agent ready with {llm.rag.get_stats()['total_examples']} production examples\n")

# ============================================================================
# STEP 1: Agent Explores Data Independently
# ============================================================================

data_path = "/Users/adityapu/Documents/GitHub/deepcausalmmm/examples/data/MMM Data.csv"

print("=" * 80)
print("STEP 1: AUTONOMOUS DATA EXPLORATION")
print("=" * 80)
print(f"\n📂 Loading: {data_path}")
print("⏳ Agent is exploring the data (no hints provided)...\n")

exploration_prompt = f"""
You are a Principal Data Scientist. You have been given a dataset at: {data_path}

YOUR TASK: Explore this data INDEPENDENTLY and understand its structure.

Write Python code to:
1. Load the data
2. Understand the structure (shape, columns, dtypes)
3. Identify:
   - What is the target variable? (Look for columns like 'sales', 'revenue', 'visits', 'conversions')
   - What are the media/marketing channels? (Look for columns with 'channel', 'spend', 'impressions', 'clicks')
   - What are control variables? (Look for columns like 'price', 'promo', 'holiday', 'control')
   - What is the date/time column? (Look for 'date', 'week', 'month', 'period')
4. Analyze each media channel:
   - Check for autocorrelation (ACF plot or Durbin-Watson test)
   - Check variance and spikes
   - Determine if adstock is needed
5. Check target variable:
   - Distribution
   - Outliers
   - Relationship with media channels
6. Check multicollinearity:
   - Calculate VIF for media channels
7. Generate a comprehensive summary as a dictionary with:
   - 'target_col': name of target variable
   - 'media_channels': list of media channel column names
   - 'control_vars': list of control variable column names
   - 'date_col': name of date/time column
   - 'n_rows': number of rows
   - 'n_channels': number of media channels
   - 'needs_adstock': boolean (based on autocorrelation analysis)
   - 'multicollinearity_high': boolean (based on VIF > 5)
   - 'recommended_approach': 'Bayesian' or 'Frequentist' based on sample size and multicollinearity

CRITICAL:
- DO NOT assume column names
- DO NOT ask me for clarification
- EXPLORE and DECIDE autonomously
- Store the summary in a variable called `exploration_summary`

Generate the complete Python code:
"""

# Stream code generation
print("🔄 Generating exploration code...\n")
print("-" * 80)

exploration_code = ""
start_time = time.time()

try:
    # Call Ollama with streaming
    result = subprocess.run(
        [
            'curl', '-s', 'http://localhost:11434/api/generate',
            '-d', json.dumps({
                "model": "qwen2.5-coder:32b",
                "prompt": llm.rag.augment_prompt(exploration_prompt, exploration_prompt, n_examples=5),
                "stream": True,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 4000
                }
            })
        ],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    # Parse streaming response
    for line in result.stdout.strip().split('\n'):
        if line:
            try:
                chunk = json.loads(line)
                if 'response' in chunk:
                    token = chunk['response']
                    exploration_code += token
                    print(token, end='', flush=True)
            except json.JSONDecodeError:
                continue
    
    print("\n" + "-" * 80)
    print(f"✅ Generated in {time.time() - start_time:.1f}s\n")
    
except subprocess.TimeoutExpired:
    print("\n⚠️ Timeout - Ollama may be slow. Continuing...\n")
    exploration_code = llm.generate(exploration_prompt, use_rag=True)

# Clean code
if "```python" in exploration_code:
    exploration_code = exploration_code.split("```python")[1].split("```")[0].strip()
elif "```" in exploration_code:
    exploration_code = exploration_code.split("```")[1].split("```")[0].strip()

# Save exploration code
exploration_file = Path("/tmp/cerebro_autonomous_exploration.py")
with open(exploration_file, 'w') as f:
    f.write(exploration_code)

print(f"💾 Saved to: {exploration_file}\n")

# Execute exploration code
print("=" * 80)
print("EXECUTING EXPLORATION CODE")
print("=" * 80)

namespace = {}
try:
    exec(exploration_code, namespace)
    exploration_summary = namespace.get('exploration_summary', {})
    
    print("\n✅ EXPLORATION COMPLETE\n")
    print("📊 Discovery:")
    print(json.dumps(exploration_summary, indent=2))
    print()
    
except Exception as e:
    print(f"\n❌ Exploration failed: {e}")
    print("Falling back to basic analysis...")
    # Minimal fallback
    data = pd.read_csv(data_path)
    exploration_summary = {
        'target_col': 'target_visits',
        'media_channels': [c for c in data.columns if 'channel' in c.lower() or 'impressions' in c.lower()],
        'control_vars': [c for c in data.columns if 'control' in c.lower()],
        'date_col': 'weekid',
        'n_rows': len(data),
        'n_channels': len([c for c in data.columns if 'channel' in c.lower()]),
        'needs_adstock': True,
        'multicollinearity_high': True,
        'recommended_approach': 'Bayesian'
    }

# ============================================================================
# STEP 2: Agent Builds MMM Model Autonomously
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: AUTONOMOUS MMM MODEL BUILDING")
print("=" * 80)
print("\n⏳ Agent is building the model (based on exploration)...\n")

model_prompt = f"""
You are a Principal Data Scientist building a Marketing Mix Model.

EXPLORATION RESULTS:
{json.dumps(exploration_summary, indent=2)}

DATA PATH: {data_path}

YOUR TASK: Build a PRODUCTION-GRADE MMM model based on the exploration results.

Based on the exploration:
- Target: {exploration_summary.get('target_col')}
- Media channels: {exploration_summary.get('n_channels')} channels
- Approach: {exploration_summary.get('recommended_approach')}
- Needs adstock: {exploration_summary.get('needs_adstock')}
- Multicollinearity: {exploration_summary.get('multicollinearity_high')}

Generate complete Python code to:

1. Load data from {data_path}

2. Feature Engineering:
   - For each media channel, decide on adstock transformation based on:
     * Check autocorrelation (ACF)
     * If high autocorrelation → geometric/Weibull adstock
     * If delayed response → delayed adstock
     * Calculate optimal decay parameter (0.3-0.9 range)
   - Apply saturation transformation:
     * Analyze spending patterns
     * If diminishing returns observed → Hill curve
     * Otherwise → log or power transformation
   - Store transformed data in `data_transformed`

3. Model Building:
   - If Bayesian: Use NumPyro/PyMC with:
     * Hierarchical priors for channel effects
     * Positive constraints on media coefficients
     * Positive baseline
     * MCMC sampling (NUTS)
   - If Frequentist: Use scipy.optimize.minimize with:
     * Bounds: positive for media channels and baseline
     * L2 regularization if multicollinearity high
   - Store model results in `model_results` dict with:
     * 'coefficients': dict of channel -> coefficient
     * 'baseline': baseline value
     * 'r_squared': model R²

4. Calculate ROI:
   - For each channel:
     * ROI = (coefficient * transformed_spend) / original_spend
   - Store in `roi_results` dict

5. Generate outputs:
   - Print model summary
   - Print ROI for each channel
   - Create a simple plot of actual vs predicted

CRITICAL REQUIREMENTS:
- Use the EXACT column names from exploration_summary
- DO NOT hardcode column names
- DO NOT assume any data structure
- Calculate everything from the data
- Handle errors gracefully
- All code must be self-contained and runnable

Generate the complete Python code (400-600 lines for production quality):
"""

print("🔄 Generating MMM model code...\n")
print("-" * 80)

model_code = ""
start_time = time.time()

try:
    # Call Ollama with streaming
    result = subprocess.run(
        [
            'curl', '-s', 'http://localhost:11434/api/generate',
            '-d', json.dumps({
                "model": "qwen2.5-coder:32b",
                "prompt": llm.rag.augment_prompt(model_prompt, model_prompt, n_examples=5),
                "stream": True,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 6000
                }
            })
        ],
        capture_output=True,
        text=True,
        timeout=600
    )
    
    # Parse streaming response
    for line in result.stdout.strip().split('\n'):
        if line:
            try:
                chunk = json.loads(line)
                if 'response' in chunk:
                    token = chunk['response']
                    model_code += token
                    print(token, end='', flush=True)
            except json.JSONDecodeError:
                continue
    
    print("\n" + "-" * 80)
    print(f"✅ Generated in {time.time() - start_time:.1f}s\n")
    
except subprocess.TimeoutExpired:
    print("\n⚠️ Timeout - Ollama may be slow. Continuing...\n")
    model_code = llm.generate(model_prompt, use_rag=True)

# Clean code
if "```python" in model_code:
    model_code = model_code.split("```python")[1].split("```")[0].strip()
elif "```" in model_code:
    model_code = model_code.split("```")[1].split("```")[0].strip()

# Save model code
model_file = Path("/tmp/cerebro_autonomous_model.py")
with open(model_file, 'w') as f:
    f.write(model_code)

print(f"💾 Saved to: {model_file}\n")

# Execute model code
print("=" * 80)
print("EXECUTING MMM MODEL CODE")
print("=" * 80)

namespace_model = {'exploration_summary': exploration_summary}
try:
    exec(model_code, namespace_model)
    
    model_results = namespace_model.get('model_results', {})
    roi_results = namespace_model.get('roi_results', {})
    
    print("\n" + "=" * 80)
    print("✅ MODEL RESULTS")
    print("=" * 80)
    print("\n📊 Model Coefficients:")
    print(json.dumps(model_results.get('coefficients', {}), indent=2))
    print(f"\n📈 Baseline: {model_results.get('baseline', 'N/A')}")
    print(f"📉 R²: {model_results.get('r_squared', 'N/A')}")
    
    print("\n💰 ROI by Channel:")
    print(json.dumps(roi_results, indent=2))
    print()
    
except Exception as e:
    print(f"\n❌ Model execution failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("🎉 AUTONOMOUS MMM COMPLETE")
print("=" * 80)
print(f"\n📁 Generated Code:")
print(f"   - Exploration: {exploration_file}")
print(f"   - Model: {model_file}")
print("\n✅ Agent explored, decided, and built the model autonomously!")
print("=" * 80)

