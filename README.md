# Cerebro: Autonomous Product and Marketing DS

**Cerebro** is an autonomous, multi-agent system that generates production-grade data science code for product and marketing analytics—no templates, no hardcoding, just intelligent code generation.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-orange.svg)](https://github.com/google/jax)
[![NumPyro](https://img.shields.io/badge/NumPyro-Bayesian-green.svg)](https://num.pyro.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Cerebro** is a fully agentic Marketing Mix Modeling (MMM) system that autonomously generates, validates, and executes high-quality Bayesian models. Built with AI agents that reason, self-heal, and adapt.

## What Makes Cerebro Different

- **Fully Agentic**: No hardcoded templates - AI generates all code from data
- **Self-Validating**: Multi-layer validation with automatic error fixing (15 retries)
- **Self-Healing**: Execution validator fixes runtime errors using RAG + few-shot learning
- **Robust**: MCMC/NUTS for Bayesian inference (like Google Meridian)
- **Backend-Aware**: Adapts to NumPyro MCMC, SVI, or JAX optimization
- **Zero Hardcoding**: All configuration from spec, all code from AI

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/cerebro.git
cd cerebro

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Credentials

Choose one of three methods:

**Option A: Config File (Recommended)**
```bash
cp .api_config.yaml.example .api_config.yaml
# Edit .api_config.yaml with your API key
```

**Option B: Environment Variable**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

**Option C: Command Line**
```bash
python3 run_pipeline.py --api-key "sk-your-key-here"
```

### 3. Run the Pipeline

```bash
# Basic usage (uses config file)
python3 run_pipeline.py

# With custom data
python3 run_pipeline.py --data-path "your_data.csv"

# With explicit credentials
python3 run_pipeline.py \
  --api-key "sk-..." \
  --base-url "https://api.openai.com/v1" \
  --data-path "your_data.csv"
```

### 4. What Happens Next

The system will:
1. **Analyze your data** - Detect channels, controls, date columns
2. **Generate spec** - Create MMMSpec YAML configuration
3. **Generate code** - Create 6 modules with AI (exploration, preprocessing, modeling, etc.)
4. **Validate code** - Multi-layer validation with self-healing
5. **Execute models** - Run Bayesian MCMC inference
6. **Create visualizations** - Generate plots and diagnostics

**Total Time**: ~30-45 minutes (mostly MCMC sampling)

## Features

### Agentic System
- **SpecWriterAgent**: Analyzes data, generates MMM specifications
- **ModelingAgent**: Creates Bayesian models with transformations
- **ValidationAgents**: Multi-layer validation (static → API → JAX → execution)
- **Self-Healing**: Automatically fixes errors using RAG + LLM reasoning

### Bayesian Modeling
- **MCMC/NUTS**: Full posterior uncertainty quantification
- **NumPyro SVI**: Scalable variational inference
- **JAX Optimization**: Fast MAP estimation
- **Transformations**: Geometric adstock, Hill saturation
- **Data-Driven Priors**: Automatic prior scaling

### Run & Config
- **Multi-Tier Credentials**: CLI args → env vars → config file
- **Backend Detection**: Adapts to numpyro_nuts, numpyro_svi, jax_optim
- **Parallel MCMC**: Multi-chain sampling for faster convergence
- **RAG Integration**: 4,224+ MMM examples
- **Comprehensive Docs**: All context in CEREBRO_COMPLETE_DOCUMENTATION.md

## Architecture

```
User Data (CSV)
    ↓
SpecWriterAgent → Analyzes data → Generates MMMSpec
    ↓
Pipeline Agents (6 modules)
    ↓
├─ DataExplorationAgent → EDA code
├─ PreprocessingAgent → Data prep code
├─ ModelingAgent → Bayesian model code
│   ├─ HybridValidator
│   │   ├─ Static validation
│   │   ├─ API validation
│   │   ├─ JAX tracing
│   │   └─ Execution validation (self-healing)
│   └─ Executes MCMC/NUTS
├─ DiagnosticsAgent → Model diagnostics
├─ OptimizationAgent → Budget optimization
└─ VisualizationAgent → Plots & charts
    ↓
Results (PKL, plots, reports)
```

## Documentation

- **[CEREBRO_COMPLETE_DOCUMENTATION.md](CEREBRO_COMPLETE_DOCUMENTATION.md)** - Complete system documentation (24 KB, 806 lines)
- **[SECURITY.md](SECURITY.md)** - Credential management guide
- **[USAGE.md](USAGE.md)** - Quick reference guide

## Configuration

### Data Requirements

Your CSV should have:
- **Date column** (any frequency: daily, weekly, monthly)
- **Target column** (outcome variable: sales, visits, revenue)
- **Channel columns** (media spend or impressions with keywords: "spend", "impression", "cost")
- **Control variables** (optional: price, promotions, holidays)

**Example:**
```csv
Date,target_visits,impressions_Channel_01,impressions_Channel_02,price,promotion
2024-01-01,1000,50000,30000,10.99,0
2024-01-02,1200,55000,32000,10.99,1
...
```

### Spec Configuration

The system auto-generates `spec_auto.yaml`:

```yaml
outcome: target_visits
date_column: Date
channels:
  - name: impressions_Channel_01
    transform:
      adstock:
        type: geometric
        max_lag: 6
      saturation:
        type: hill
controls:
  - price
  - promotion
inference:
  backend: numpyro_nuts  # or numpyro_svi, jax_optim
  num_warmup: 500
  num_samples: 500
  num_chains: 1
```

## Example Queries

### Check Model Results
```python
import pickle
with open('module_3_results.pkl', 'rb') as f:
    results = pickle.load(f)

print(f"R²: {results['r2']:.4f}")
print(f"Channels: {results['channel_names']}")
print(f"Coefficients: {results['params']}")
```

### Customize MCMC
Edit `spec_auto.yaml`:
```yaml
inference:
  backend: numpyro_nuts
  num_warmup: 1000      # More warmup for complex models
  num_samples: 1000     # More samples for better posteriors
  num_chains: 4         # Parallel chains (requires numpyro.set_host_device_count(4))
```

### Use Different Backend
```yaml
inference:
  backend: numpyro_svi  # Faster approximate inference
  num_steps: 10000
  learning_rate: 0.01
```

## Advanced Usage

### Custom Credential Method

```python
from cerebro.llm.api_backend import ApiBackend

# Direct instantiation
llm = ApiBackend(
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
    model="gpt-4o"
)
```

### Run Specific Modules

```python
from cerebro.spec.schema import MMMSpec
from cerebro.llm.api_backend import ApiBackend
from cerebro.agents.modeling_agent import ModelingAgent

# Load spec
spec = MMMSpec.from_yaml('spec_auto.yaml')

# Initialize agent
llm = ApiBackend()
modeling_agent = ModelingAgent(llm, use_rag=True)

# Generate model code
model_code = modeling_agent.generate_model_code(
    spec=spec,
    data_path='preprocessed_data.csv',
    enable_validation=True
)

# Execute
exec_globals = {}
exec(model_code, exec_globals)
results = exec_globals['run_modeling']('preprocessed_data.csv')
```

### Add Custom Transformations

The system auto-generates transformations based on spec, but you can customize:

```yaml
channels:
  - name: TV
    transform:
      adstock:
        type: geometric  # or: delayed, carryover, exponential
        max_lag: 8       # Custom lag
      saturation:
        type: hill       # or: logistic, exponential
```

## Performance

**Validation Speed:**
- Static validation: <1 second
- Execution validation: ~30 seconds (minimal MCMC: 10 warmup + 10 samples)
- Full MCMC: ~20-30 minutes (500 warmup + 500 samples × 1 chain)

**Model Quality:**
- Typical R²: 0.95-0.99 on real MMM data
- Acceptance rate: 0.85-0.95 (well-conditioned)
- Divergences: <1% (healthy MCMC)

**System Stats:**
- Code generation: ~15 seconds per module
- Total pipeline: ~30-45 minutes end-to-end
- Self-healing success: >90% within 15 retries

## Testing

```bash
# Test core imports
python3 -c "
from cerebro.agents.spec_writer_agent import AutonomousSpecWriter
from cerebro.agents.modeling_agent import ModelingAgent
from cerebro.llm.api_backend import ApiBackend
print('All imports successful!')
"

# Run full pipeline (test mode)
python3 run_pipeline.py --data-path "examples/MMM Data.csv"
```

## Troubleshooting

### Slow MCMC (1023 leapfrog steps)

**Symptom**: Each sample takes 15-20 seconds

**Cause**: Poorly conditioned posterior

**Fix**: The system automatically scales priors, but if issues persist:
```python
# In generated model code, priors are already scaled by sqrt(n_channels)
# If still slow, try:
# 1. Use LogNormal(0, 0.5) for transformation parameters
# 2. Scale data to [0, 1]
# 3. Reduce num_chains to 1
```

### NaN Loss

**Symptom**: RuntimeError: NaN detected in loss

**Cause**: Exploding gradients or poor priors

**Fix**: System auto-fixes this, but manual fix:
```python
# System automatically:
# 1. Scales inputs to [0, 1]
# 2. Uses data-driven priors
# 3. Applies gradient clipping
```

### API Credentials Not Found

**Symptom**: ValueError: API key not found

**Fix**:
```bash
# Option 1: Config file
cp .api_config.yaml.example .api_config.yaml
# Edit with your key

# Option 2: Environment variable
export OPENAI_API_KEY="sk-..."

# Option 3: Command line
python3 run_pipeline.py --api-key "sk-..."
```

## Project Structure

```
cerebro/
├── README.md                              ← You are here
├── CEREBRO_COMPLETE_DOCUMENTATION.md     ← Full system docs
├── SECURITY.md                            ← Credential management
├── USAGE.md                               ← Quick reference
├── .api_config.yaml.example              ← Config template
├── run_pipeline.py                   ← Main pipeline
├── cerebro/
│   ├── agents/                            ← AI agents
│   │   ├── spec_writer_agent.py          ← Spec generation
│   │   ├── modeling_agent.py             ← Model generation
│   │   ├── data_exploration_agent.py     ← EDA
│   │   ├── preprocessing_agent.py        ← Data prep
│   │   ├── diagnostics_agent.py          ← Model diagnostics
│   │   ├── optimization_agent.py         ← Budget optimization
│   │   └── visualization_agent.py        ← Plotting
│   ├── llm/                               ← LLM backends
│   │   └── api_backend.py                ← OpenAI/compatible APIs
│   ├── spec/                              ← Schema definitions
│   │   └── schema.py                     ← MMMSpec Pydantic model
│   └── utils/                             ← Utilities
└── examples/                              ← Example scripts
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **NumPyro**: Bayesian inference framework
- **JAX**: Accelerated computing
- **Google Meridian**: Inspiration for MCMC architecture
- **PyMC-Marketing**: MMM best practices
- **OpenAI**: GPT-4o for agentic reasoning

## Support

- **Documentation**: See [CEREBRO_COMPLETE_DOCUMENTATION.md](CEREBRO_COMPLETE_DOCUMENTATION.md)
- **Issues**: [GitHub Issues](https://github.com/yourusername/cerebro/issues)
- **Email**: your.email@example.com

---

**Built with AI for MMM practitioners**

