# Security: Credentials and Secrets

## Credential sources (priority order)

1. **Command line** – `run_pipeline.py --api-key "sk-..." --base-url "https://..."`
2. **Environment** – `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`
3. **Config file** – `.api_config.yaml` (see below)

## Config file

```bash
cp .api_config.yaml.example .api_config.yaml
# Edit .api_config.yaml with your API key; do not commit this file.
```

Confirm it is ignored:

```bash
git check-ignore .api_config.yaml
# Should output: .api_config.yaml
```

## What is gitignored

- `.api_config.yaml` – your real API credentials
- `*.log` – log files
- `*.pkl` – model outputs
- Generated files: `module_*.py`, `module_*.pkl`, `spec_auto*.yaml`, `module_3_results.json`, etc. (see `.gitignore`)

## Safe to commit

- Source code under `cerebro/`
- `.api_config.yaml.example` (no real keys)
- `README.md`, `SECURITY.md`, `LICENSE`

## If you commit credentials

1. Rotate or revoke the key immediately.
2. Remove from history (e.g. `git filter-repo` or BFG) and force-push if necessary.
3. Create new credentials.

## File permissions

```bash
chmod 600 .api_config.yaml
```
