# 🧠 Cerebro: Autonomous Marketing Mix Modeling

**Cerebro** is an autonomous, multi-agent system that generates production-grade Marketing Mix Model (MMM) code from your data—no templates, no hardcoding, just intelligent code generation.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What It Does

```
Your CSV Data → Cerebro → Production MMM Pipeline (1000+ lines of Python)
```

**Cerebro autonomously:**
1. 📊 Analyzes your data (channels, outcomes, controls, seasonality)
2. 🎨 Designs a custom MMM spec (transformations, priors, inference)
3. 💻 Generates production code (data exploration, modeling, diagnostics, optimization)
4. 🚀 Delivers a complete, executable pipeline

**No templates. No hardcoded assumptions. Just intelligent, data-driven code generation.**

---

## ✨ Key Features

- **🤖 Fully Autonomous**: Understands your data structure and generates appropriate code
- **🎯 Multi-Agent Architecture**: Specialized agents for exploration, modeling, diagnostics, optimization
- **📚 RAG-Powered**: 4,000+ production examples from Google, Meta, Uber, Microsoft MMM repos
- **🔧 Production-Ready**: Generates 1000+ lines of detailed, documented code
- **🎨 Flexible Backends**: NumPyro (JAX), PyMC, or Stan
- **⚡ Fast**: Local LLM (Qwen) via Ollama or API models (Claude, GPT-4)
- **📝 Declarative Specs**: YAML-based model specifications for reproducibility

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cerebro.git
cd cerebro

# Install dependencies
pip install -e .

# Install Ollama (for local LLM)
# macOS:
brew install ollama

# Linux:
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama and pull model
ollama serve
ollama pull qwen2.5:7b
```

### 2. Run Autonomous MMM

```bash
# Generate complete MMM pipeline from your data
python -m cerebro auto path/to/your/data.csv --output pipeline.py
```

**That's it!** Cerebro will:
- Analyze your data
- Generate a model spec
- Write 1000+ lines of production Python
- Save everything to `pipeline.py` and `mmm_spec.yaml`

### 3. Use the Generated Code

```bash
# Run the generated pipeline
python pipeline.py
```

---

## 📖 Examples

### Example 1: Autonomous Generation

```python
from cerebro.cli import auto_mmm

# Generate complete MMM from data
auto_mmm(
    data_path="data/mmm_data.csv",
    output_path="generated_mmm.py"
)
```

### Example 2: From Spec

```python
from cerebro.cli import generate_from_spec

# Generate code from existing spec
generate_from_spec(
    spec_path="mmm_spec.yaml",
    output_path="mmm_pipeline.py"
)
```

### Example 3: Custom RAG Query

```python
from cerebro.llm import RAGBackend

rag = RAGBackend()
examples = rag.search("NumPyro SVI with autoguide", n_results=5)
print(examples)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CEREBRO ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📊 Data → SpecWriterAgent → YAML Spec                         │
│                                                                 │
│  YAML Spec → Multi-Agent Orchestrator                          │
│                ├─ DataExplorationAgent  → exploration.py       │
│                ├─ PreprocessingAgent    → preprocessing.py     │
│                ├─ ModelingAgent         → model.py             │
│                ├─ DiagnosticsAgent      → diagnostics.py       │
│                ├─ OptimizationAgent     → optimization.py      │
│                └─ VisualizationAgent    → visualization.py     │
│                                                                 │
│  Generated Modules → Orchestrator → Complete Pipeline          │
│                                                                 │
│  📚 RAG: 4,049 Examples (NumPyro, PyMC, JAX, Production MMMs)  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Components

1. **SpecWriterAgent**: Analyzes data → generates declarative YAML spec
2. **Multi-Agent System**: Specialized agents for each pipeline stage
3. **RAG Backend**: 4,049 production examples from top MMM libraries
4. **Orchestrator**: Combines modules into complete executable pipeline
5. **LLM Backends**: Local (Ollama/MLX) or API (Claude, GPT-4)

---

## 📊 What Code Does It Generate?

Cerebro generates a **complete, production-grade MMM pipeline** (~1000-1500 lines):

### 1. Data Exploration (200-300 lines)
- Descriptive statistics
- Time series analysis (ACF, PACF, stationarity)
- Correlation analysis (Pearson, Spearman, Kendall)
- Multicollinearity checks (VIF)
- Outlier detection (Z-score, IQR, Isolation Forest)

### 2. Preprocessing (200-250 lines)
- Missing value imputation
- Outlier treatment (winsorization, clipping)
- Feature engineering (lags, rolling windows, rates)
- Scaling and normalization
- Data validation

### 3. Model Building (250-350 lines)
- **Transformations**: Adstock (geometric, Weibull, delayed), Saturation (Hill, logistic)
- **Model**: Bayesian regression with priors
- **Training**: NumPyro SVI, PyMC NUTS, or Stan sampling
- **Inference**: Posterior sampling and predictions

### 4. Diagnostics (100-150 lines)
- Convergence checks (R-hat, ESS)
- Posterior predictive checks
- Model comparison (LOO, WAIC)
- Residual analysis

### 5. Budget Optimization (100-150 lines)
- ROI curve calculation
- Optimal budget allocation
- Marginal ROI analysis
- Scenario testing

### 6. Visualization (100-150 lines)
- Channel contribution waterfall
- Response curves
- Time series decomposition
- Diagnostic plots

---

## 🎓 RAG Knowledge Base

Cerebro's RAG system contains **4,049 production examples** from:

| Source | Examples | Description |
|--------|----------|-------------|
| **JAX** | 1,525 | Shape operations, broadcasting, jit, vmap |
| **PyMC** | 268 | Bayesian modeling API, sampling |
| **NumPyro** | 66 | SVI, autoguide, distributions |
| **PyMC-Marketing** | 308 | Production Bayesian MMM |
| **Meridian (Google)** | 249 | Advanced Bayesian MMM |
| **LightweightMMM (Google)** | 41 | JAX-based MMM |
| **Robyn (Meta)** | 160 | Meta's MMM framework |
| **EconML (Microsoft)** | 376 | Causal inference |
| **CausalML (Uber)** | 120 | Uplift modeling |
| **Others** | 936 | Prophet, Orbit, Kats, DoWhy, etc. |

---

## 🔧 Configuration

### Using Local LLM (Ollama)

```bash
# Default: Uses Ollama with Qwen 2.5 7B
cerebro auto data.csv
```

### Using API Models

```bash
# Claude Sonnet (recommended for production)
export ANTHROPIC_API_KEY="your-key"
cerebro auto data.csv --llm claude

# OpenAI GPT-4
export OPENAI_API_KEY="your-key"
cerebro auto data.csv --llm gpt4
```

### Custom Spec

```yaml
# mmm_spec.yaml
name: "Q4 Campaign Analysis"
outcome: "conversions"
date_column: "week"
channels:
  - name: "tv_impressions"
    adstock:
      type: "weibull"
      max_lag: 8
    saturation:
      type: "hill"
controls:
  - "price_index"
  - "competitor_spend"
inference:
  backend: "numpyro_svi"
  iterations: 50000
```

```bash
cerebro generate mmm_spec.yaml --output pipeline.py
```

---

## 📁 Project Structure

```
cerebro/
├── cerebro/
│   ├── agents/           # Multi-agent system
│   │   ├── spec_writer_agent.py
│   │   ├── data_exploration_agent.py
│   │   ├── modeling_agent.py
│   │   └── ...
│   ├── llm/              # LLM backends & RAG
│   │   ├── auto_backend.py
│   │   ├── rag_backend.py
│   │   └── ...
│   ├── spec/             # YAML spec schema
│   └── cli.py            # Command-line interface
├── examples/             # Usage examples
├── fine_tuning/          # RAG datasets & training
└── tests/                # Test suite
```

---

## 🧪 Development

### Running Tests

```bash
pytest tests/
```

### Rebuilding RAG Index

```bash
# Download production MMM code + API examples
python fine_tuning/rebuild_rag_with_api_examples.py

# Rebuild ChromaDB index
python rebuild_rag_index.py
```

### Adding New Agents

```python
from cerebro.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def generate_code(self, spec):
        # Your logic here
        return generated_code
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

**Built on the shoulders of giants:**

- [NumPyro](https://github.com/pyro-ppl/numpyro) - Probabilistic programming with JAX
- [PyMC](https://github.com/pymc-devs/pymc) - Bayesian modeling
- [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) - MMM with PyMC
- [LightweightMMM](https://github.com/google/lightweight_mmm) - Google's JAX MMM
- [Robyn](https://github.com/facebookexperimental/Robyn) - Meta's MMM
- [Meridian](https://github.com/google/meridian) - Google's Bayesian MMM
- [Ollama](https://ollama.ai) - Local LLM serving
- [ChromaDB](https://www.trychroma.com/) - Vector database for RAG

---

## 📧 Contact

Questions? Issues? Ideas?

- **GitHub Issues**: [github.com/yourusername/cerebro/issues](https://github.com/yourusername/cerebro/issues)
- **Email**: your.email@example.com

---

## 🎯 Roadmap

- [ ] Support for hierarchical models (geo, product hierarchy)
- [ ] Automated experiment calibration
- [ ] Interactive Streamlit UI
- [ ] Fine-tuned local models
- [ ] MLflow integration for tracking
- [ ] Docker deployment
- [ ] Cloud deployment templates (AWS, GCP, Azure)

---

**Made with ❤️ for data scientists who want to focus on insights, not implementation.**
