# Fine-Tuning & RAG Data

## 📊 RAG Database

**Size**: 4,049 examples from production MMM repositories + API examples  
**Content**: NumPyro, PyMC, JAX, LightweightMMM, Meridian, Robyn, PyMC-Marketing, etc.

### Getting the RAG Database

**Option 1: Build from Scratch (Recommended)**
```bash
# Rebuilds RAG by cloning repos and extracting examples
python fine_tuning/rebuild_rag_with_api_examples.py
python rebuild_rag_index.py
```
This will:
- Clone NumPyro, PyMC, JAX repositories (~1GB download)
- Clone production MMM repos (~2GB download)  
- Extract 4,049 relevant code examples
- Build ChromaDB index

**Option 2: Download Pre-built (Coming Soon)**
```bash
# Download pre-built RAG database (18MB)
wget https://github.com/yourusername/cerebro/releases/download/v0.1.0/rag_production_plus_api.jsonl
# Index it
python rebuild_rag_index.py --dataset fine_tuning/rag_production_plus_api.jsonl
```

> **Note**: The RAG database and ChromaDB index are excluded from git due to size (87MB indexed).  
> First-time users should run Option 1 to build the RAG system.

## 🔧 Rebuilding RAG

If you want to rebuild the RAG database:

```bash
# 1. Run the rebuild script (clones repos, extracts examples)
python fine_tuning/rebuild_rag_with_api_examples.py

# 2. Rebuild ChromaDB index
python rebuild_rag_index.py
```

**Note**: Rebuilding downloads ~3GB of source repositories. The script will:
- Clone NumPyro, PyMC, JAX repositories
- Clone production MMM repos (PyMC-Marketing, Meridian, Robyn, etc.)
- Extract relevant code examples
- Generate new `rag_production_plus_api.jsonl`

## 📝 What's Included

- `rag_production_plus_api.jsonl` - Current RAG dataset (4,049 examples)
- `rebuild_rag_with_api_examples.py` - Script to rebuild from scratch
- `rebuild_production_rag.py` - Alternative rebuild script

## 🗑️ What's NOT Included (Too Large for GitHub)

- `mmm_sources/` (3GB) - Cloned source repositories (rebuilt on demand)
- Old training datasets (deprecated)
- Model checkpoints (use Hugging Face for distribution)
- Virtual environments

## 🚀 Quick Start

The RAG database is already built and ready to use:

```python
from cerebro.llm import RAGBackend

rag = RAGBackend()
examples = rag.search("NumPyro SVI example", n_results=5)
```

No rebuild needed unless you want to add more examples!
