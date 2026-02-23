# Security Guidelines

## Credential Management

This project uses a secure credential management system to ensure API keys and sensitive information are never committed to git.

### Quick Start

1. **Copy the example config:**
   ```bash
   cp .api_config.yaml.example .api_config.yaml
   ```

2. **Add your credentials:**
   Edit `.api_config.yaml` with your actual API key:
   ```yaml
   api:
     provider: "openai"
     base_url: "https://api.openai.com/v1"
     api_key: "sk-your-actual-key-here"
     model: "gpt-4o"
   ```

3. **Verify it's gitignored:**
   ```bash
   git check-ignore .api_config.yaml
   # Should output: .api_config.yaml
   ```

### What's Protected

✅ **Gitignored Files:**
- `.api_config.yaml` - Your actual API credentials (NEVER committed)
- `*.log` - Log files that might contain API responses
- `*.pkl` - Model results files

✅ **No Hardcoded Credentials:**
- All source code reads credentials at runtime
- No API keys in `cerebro/llm/api_backend.py` or any other code
- No company-specific endpoints hardcoded

✅ **Example Files (Safe to Commit):**
- `.api_config.yaml.example` - Template without real credentials
- `USAGE.md` - Documentation with examples only
- `SECURITY.md` - This file

### Alternative Credential Methods

The pipeline supports 3 methods (in priority order):

1. **Command-line arguments** (highest priority)
   ```bash
   python3 run_pipeline.py --api-key "sk-..." --base-url "https://api.openai.com/v1"
   ```

2. **Environment variables**
   ```bash
   export OPENAI_API_KEY="sk-..."
   export OPENAI_BASE_URL="https://api.openai.com/v1"
   python3 run_pipeline.py
   ```

3. **Config file** (lowest priority)
   ```bash
   # Uses .api_config.yaml by default
   python3 run_pipeline.py
   ```

### Security Verification

Run this before committing:

```bash
# Check for any Zillow/company-specific references
git grep -i "zillow\|zgai\|internal" 

# Check for hardcoded API keys
git grep -E "eyJhbG|sk-[A-Za-z0-9]{32,}|api.*key.*=.*['\"]"

# Verify .api_config.yaml is gitignored
git check-ignore .api_config.yaml

# Verify no credentials are tracked
git ls-files | xargs grep -l "api.*key.*=" || echo "All clear!"
```

### What Gets Committed

✅ **Safe to commit:**
- Source code (`cerebro/`)
- Example configs (`.example` files)
- Documentation (`README.md`, `USAGE.md`, `SECURITY.md`)
- Example data (`examples/`)

❌ **Never commit:**
- `.api_config.yaml` (your real credentials)
- Log files with API responses
- Model output files with proprietary data
- Any files with actual API keys or tokens

### File Permissions

Set restrictive permissions on your config file:

```bash
chmod 600 .api_config.yaml
# Only you can read/write
```

### If Credentials Are Accidentally Committed

If you accidentally commit credentials:

1. **Immediately rotate/revoke the API key**
2. **Remove from git history:**
   ```bash
   # Use git-filter-repo or BFG Repo-Cleaner
   git filter-repo --invert-paths --path .api_config.yaml
   ```
3. **Force push (if needed):**
   ```bash
   git push origin --force --all
   ```
4. **Generate new credentials**

### Contact

For security concerns, please contact the repository maintainer.

---

**Last Updated:** December 15, 2025
**Status:** ✅ All credentials secure, no hardcoded secrets in git

