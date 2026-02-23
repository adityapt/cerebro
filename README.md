# Cerebro: Autonomous Marketing Mix Modeling

**Cerebro** is an agentic system that generates, validates, and executes Bayesian Marketing Mix Modeling (MMM) code. Agents reason over your data, produce a spec, then generate and run six modules (exploration, preprocessing, modeling, diagnostics, optimization, visualization) with no hardcoded templates.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-orange.svg)](https://github.com/google/jax)
[![NumPyro](https://img.shields.io/badge/NumPyro-Bayesian-green.svg)](https://num.pyro.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What It Does

- **Agentic spec**: Infers channels, controls, date, outcome from your CSV and produces an MMMSpec (YAML).
- **Six modules**: Data exploration, preprocessing, Bayesian modeling (NumPyro/JAX), diagnostics, optimization, visualization.
- **Validation**: HybridValidator (static, API, JAX tracing, execution) with self-healing and up to 15 retries.
- **Execution**: Generated code runs after each module; optional `--start-from diagnostics` to skip earlier steps when you have `module_3_results.json` and preprocessed data.

## Quick Start

### Install

```bash
git clone https://github.com/adityapt/cerebro.git
cd cerebro
pip install -r requirements.txt
```

### Credentials (pick one)

| Method | Usage |
|--------|--------|
| **Config file** | `cp .api_config.yaml.example .api_config.yaml` then edit with your API key |
| **Environment** | `export OPENAI_API_KEY="sk-..."` |
| **CLI** | `python3 run_pipeline.py --api-key "sk-..."` |

See [SECURITY.md](SECURITY.md) for details.

### Run pipeline

```bash
# Default: generate spec from data, then run all modules (modular mode)
python3 run_pipeline.py

# Custom data and spec path
python3 run_pipeline.py --data-path "your_data.csv" --spec-path "spec_auto.yaml"

# Resume from diagnostics (requires module_3_results.json and preprocessed CSV)
python3 run_pipeline.py --start-from diagnostics --data-path "preprocessed.csv"

# Monolithic: one generated file, no per-module execution
python3 run_pipeline.py --mode monolithic

# Disable validation or saving modules
python3 run_pipeline.py --no-validate --no-save
```

### CLI (alternative entry)

```bash
# Generate from data (uses AutoBackend: ollama/vLLM/MLX by default)
cerebro auto data.csv -o output.py --save-spec [--validate]

# Generate from existing spec
cerebro generate spec.yaml -o output.py [--data-path data.csv] [--validate]

# Validate existing code
cerebro validate code.py data.csv [-o fixed.py]
```

## Command reference (run_pipeline.py)

| Option | Default | Description |
|--------|---------|-------------|
| `--api-key` | env/config | OpenAI-compatible API key |
| `--base-url` | env/config | API base URL |
| `--model` | gpt-4o | Model name |
| `--data-path` | examples/MMM Data.csv | Input CSV |
| `--spec-path` | spec_auto.yaml | Where to save/load spec |
| `--mode` | modular | `modular` or `monolithic` |
| `--no-validate` | false | Skip validation |
| `--no-save` | false | Do not save module_*.py |
| `--log-file` | pipeline_run.log | Log path |
| `--start-from` | (none) | `diagnostics` to skip exploration/preprocessing/modeling |

## Data requirements

CSV with:

- **Date column** (daily/weekly/monthly)
- **Target column** (e.g. sales, visits, revenue)
- **Channel columns** (spend or impressions; names often contain "spend", "impression", "cost")
- **Controls** (optional: price, promotion, etc.)

## Architecture

```
CSV → SpecWriterAgent → MMMSpec (YAML)
         ↓
OrchestratorAgent (GOT reasoning)
         ↓
Modular pipeline: Exploration → Preprocessing → Modeling → Diagnostics → Optimization → Visualization
         ↓
Per module: Generate → HybridValidator (static/API/JAX/exec) → Execute
         ↓
Results (module_*.py, module_3_results.json, plots, etc.)
```

**Agents**: `spec_writer_agent`, `orchestrator_agent`, `data_exploration_agent`, `preprocessing_agent`, `modeling_agent`, `diagnostics_agent`, `optimization_agent`, `visualization_agent`. Validation: `execution_validator`, `hybrid_validator`; context: `pipeline_context`.

**LLM**: Default is `ApiBackend` (OpenAI-compatible). Optional backends: `auto_backend` (MLX/vLLM/Ollama), `ollama_backend`, `vllm_backend`, `mlx_backend`, `hybrid_backend`, `qwen_rag_backend`.

## Project structure

```
cerebro/
├── README.md
├── SECURITY.md
├── .api_config.yaml.example
├── run_pipeline.py              # Main pipeline (spec + orchestration)
├── cerebro/
│   ├── agents/
│   │   ├── spec_writer_agent.py
│   │   ├── orchestrator_agent.py
│   │   ├── execution_validator.py
│   │   ├── hybrid_validator.py
│   │   ├── pipeline_context.py
│   │   ├── base_agent.py
│   │   ├── data_exploration_agent.py
│   │   ├── preprocessing_agent.py
│   │   ├── modeling_agent.py
│   │   ├── diagnostics_agent.py
│   │   ├── optimization_agent.py
│   │   └── visualization_agent.py
│   ├── llm/
│   │   ├── api_backend.py        # OpenAI-compatible (default)
│   │   ├── auto_backend.py
│   │   ├── ollama_backend.py
│   │   ├── vllm_backend.py
│   │   ├── mlx_backend.py
│   │   ├── hybrid_backend.py
│   │   ├── tree_of_thought.py
│   │   ├── code_judge.py
│   │   └── results_judge.py
│   ├── spec/
│   │   └── schema.py             # MMMSpec
│   └── utils/
└── examples/
```

## Configuration (spec)

The pipeline writes `spec_auto.yaml`. You can edit it to change inference:

```yaml
inference:
  backend: numpyro_nuts   # or numpyro_svi, jax_optim
  num_warmup: 500
  num_samples: 500
  num_chains: 1
```

## Troubleshooting

| Issue | Action |
|-------|--------|
| API key not found | Use config file, `OPENAI_API_KEY`, or `--api-key` (see [SECURITY.md](SECURITY.md)) |
| Slow MCMC | Increase warmup; check priors; scale data |
| NaN in loss | System applies scaling and clipping; check data and spec |

## Docs

- [SECURITY.md](SECURITY.md) – Credential handling and what is gitignored.

## License

MIT. See LICENSE.
