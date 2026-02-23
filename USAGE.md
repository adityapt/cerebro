# Cerebro MMM Pipeline - Usage Guide

## Running the Pipeline

The Cerebro pipeline can be run with various credential input methods.

### Method 1: Using Config File (Default)

Copy the example config and add your credentials:

```bash
cp .api_config.yaml.example .api_config.yaml
# Edit .api_config.yaml with your API key
```

Example `.api_config.yaml`:

```yaml
api:
  provider: "openai"
  base_url: "https://api.openai.com/v1"
  api_key: "sk-your-api-key-here"
  model: "gpt-4o"
  max_tokens: 4096
```

Then run:

```bash
python3 run_pipeline.py
```

### Method 2: Using Environment Variables

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional
export OPENAI_MODEL="gpt-4o"  # Optional

python3 run_pipeline.py
```

### Method 3: Using Command Line Arguments

```bash
python3 run_pipeline.py \
  --api-key "sk-your-api-key-here" \
  --base-url "https://api.openai.com/v1" \
  --model "gpt-4o"
```

### Method 4: Custom Data Path

```bash
python3 run_pipeline.py \
  --api-key "sk-your-api-key-here" \
  --data-path "path/to/your/data.csv" \
  --spec-path "custom_spec.yaml"
```

## Credential Priority

The pipeline checks for credentials in this order:

1. **Command line arguments** (highest priority)
2. **Environment variables** (OPENAI_API_KEY, OPENAI_BASE_URL, etc.)
3. **Config file** (.api_config.yaml)

## Pipeline Output

The pipeline generates:

- `spec_auto.yaml` - Auto-generated MMM specification
- `module_exploration.py` - Data exploration code
- `module_preprocessing.py` - Data preprocessing code
- `module_modeling.py` - Bayesian MMM model code
- `module_diagnostics.py` - Model diagnostics code
- `module_optimization.py` - Budget optimization code
- `module_visualization.py` - Results visualization code
- `module_3_results.pkl` - Model parameters and results
- Various plots (PNG files)

## Example: Full Pipeline Run

```bash
# Set credentials via environment
export OPENAI_API_KEY="sk-..."

# Run with custom data
python3 run_pipeline.py \
  --data-path "examples/MMM Data.csv" \
  --spec-path "my_spec.yaml"
```

## Notes

- The pipeline automatically detects channels, controls, and date columns
- All code generation is streamed in real-time
- Validators auto-fix errors (up to 15 retries per module)
- MCMC configuration is read from the generated spec
- No emojis in output (clean, production-ready logs)

