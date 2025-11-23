"""
Enhanced RAG Builder: Production MMM Code + Package API Examples
Includes NumPyro/PyMC/JAX examples and tests for correct API usage patterns
"""

import json
import logging
from pathlib import Path
import subprocess
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clone_package_repo(package_name, git_url, target_dir):
    """Clone a package repository if not already present"""
    if target_dir.exists():
        logger.info(f"  ✓ {package_name} already cloned")
        return True
    
    try:
        logger.info(f"  Cloning {package_name}...")
        subprocess.run(
            ['git', 'clone', '--depth', '1', git_url, str(target_dir)],
            check=True,
            capture_output=True
        )
        logger.info(f"  ✓ Cloned {package_name}")
        return True
    except Exception as e:
        logger.error(f"  ✗ Failed to clone {package_name}: {e}")
        return False

def extract_api_examples(repo_path, package_name):
    """Extract working code examples from package repositories"""
    examples = []
    
    # Target directories - examples and tests contain working code
    target_dirs = ['examples', 'tests', 'tutorials', 'notebooks']
    
    # API patterns we want to capture
    api_patterns = [
        'SVI', 'autoguide', 'Predictive', 'sample', 'numpyro.sample',
        'jax.jit', 'jax.vmap', 'jax.lax', 'jnp.',
        'pm.Model', 'pm.sample', 'pm.Deterministic',
        'dist.Normal', 'dist.Beta', 'dist.Gamma',
        'shape', 'reshape', 'broadcast', 'expand_dims',
        'convolve', 'scan', 'fori_loop',
        'Trace_ELBO', 'Adam', 'RMSProp',
        'guide', 'model', 'inference'
    ]
    
    for target_dir in target_dirs:
        search_path = repo_path / target_dir
        if not search_path.exists():
            continue
        
        logger.info(f"    Scanning {package_name}/{target_dir}...")
        
        for file_path in search_path.rglob('*.py'):
            # Skip __pycache__ and other noise
            if '__pycache__' in str(file_path) or file_path.name.startswith('_'):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Must be substantial and contain API patterns
                if len(content) < 300:
                    continue
                
                content_lower = content.lower()
                pattern_count = sum(1 for pattern in api_patterns if pattern.lower() in content_lower)
                
                if pattern_count < 2:
                    continue
                
                # Chunk if too large
                if len(content) > 8000:
                    chunks = []
                    lines = content.split('\n')
                    current_chunk = []
                    current_size = 0
                    
                    for line in lines:
                        current_chunk.append(line)
                        current_size += len(line) + 1
                        
                        if current_size >= 4000:
                            chunk_content = '\n'.join(current_chunk)
                            # Only keep chunks with API patterns
                            if sum(1 for p in api_patterns if p in chunk_content) >= 2:
                                chunks.append(chunk_content)
                            current_chunk = []
                            current_size = 0
                    
                    if current_chunk:
                        chunk_content = '\n'.join(current_chunk)
                        if sum(1 for p in api_patterns if p in chunk_content) >= 2:
                            chunks.append(chunk_content)
                    
                    for i, chunk in enumerate(chunks):
                        rel_path = file_path.relative_to(repo_path)
                        examples.append({
                            'instruction': f'{package_name} API example: {rel_path} (part {i+1}/{len(chunks)})',
                            'output': chunk,
                            'repo': package_name,
                            'file': str(rel_path),
                            'type': 'api_example'
                        })
                else:
                    # Single example
                    rel_path = file_path.relative_to(repo_path)
                    examples.append({
                        'instruction': f'{package_name} API example: {rel_path}',
                        'output': content,
                        'repo': package_name,
                        'file': str(rel_path),
                        'type': 'api_example'
                    })
                
                logger.debug(f"      ✓ {rel_path} ({len(content)} chars)")
                
            except Exception as e:
                logger.warning(f"      ✗ Failed to read {file_path}: {e}")
                continue
    
    return examples

def extract_production_examples(repo_path, repo_name):
    """Extract production MMM code (original function)"""
    examples = []
    
    include_patterns = ['**/*.py']
    skip_patterns = [
        'test_', '_test.py', 'conftest', '__init__.py', 'setup.py',
        '__pycache__', '.pyc', 'example.py', 'demo.py', 'tutorial',
        'docs/', 'doc/', 'notebooks/', 'examples/'
    ]
    
    for pattern in include_patterns:
        for file_path in repo_path.rglob(pattern):
            if any(skip in str(file_path) for skip in skip_patterns):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if len(content) < 200:
                    continue
                
                # Chunk large files
                if len(content) > 10000:
                    chunks = []
                    lines = content.split('\n')
                    current_chunk = []
                    current_size = 0
                    
                    for line in lines:
                        current_chunk.append(line)
                        current_size += len(line) + 1
                        
                        if current_size >= 5000:
                            chunks.append('\n'.join(current_chunk))
                            current_chunk = []
                            current_size = 0
                    
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                    
                    for i, chunk in enumerate(chunks):
                        if len(chunk) > 500:
                            rel_path = file_path.relative_to(repo_path)
                            examples.append({
                                'instruction': f'Production MMM code from {repo_name}: {rel_path} (part {i+1}/{len(chunks)})',
                                'output': chunk,
                                'repo': repo_name,
                                'file': str(rel_path),
                                'type': 'production_code'
                            })
                    continue
                
                # Keyword filter
                relevant_keywords = [
                    'adstock', 'saturation', 'transform', 'hill', 'weibull',
                    'geometric', 'delayed', 'carryover', 'decay',
                    'mmm', 'marketing', 'media', 'channel', 'spend',
                    'bayesian', 'pymc', 'numpyro', 'stan', 'jax',
                    'likelihood', 'prior', 'posterior', 'mcmc',
                    'regression', 'coefficient', 'roi', 'roas', 'response',
                    'attribution', 'contribution', 'effectiveness',
                    'def fit', 'def train', 'def predict', 'def optimize',
                    'model', 'loss', 'gradient', 'parameter'
                ]
                
                content_lower = content.lower()
                keyword_count = sum(1 for kw in relevant_keywords if kw in content_lower)
                if keyword_count < 2:
                    continue
                
                rel_path = file_path.relative_to(repo_path)
                examples.append({
                    'instruction': f'Production MMM code from {repo_name}: {rel_path}',
                    'output': content,
                    'repo': repo_name,
                    'file': str(rel_path),
                    'type': 'production_code'
                })
                
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                continue
    
    return examples

def main():
    logger.info("="*80)
    logger.info("REBUILDING RAG: PRODUCTION CODE + API EXAMPLES")
    logger.info("="*80)
    logger.info("")
    
    sources_dir = Path('fine_tuning/mmm_sources')
    sources_dir.mkdir(parents=True, exist_ok=True)
    
    all_examples = []
    
    # =========================================================================
    # PART 1: Production MMM Repositories
    # =========================================================================
    logger.info("PART 1: Extracting Production MMM Code")
    logger.info("-" * 80)
    
    mmm_repos = {
        'pymc-marketing': 'PyMC Marketing (Bayesian MMM)',
        'lightweight_mmm': 'LightweightMMM (Google JAX MMM)',
        'robyn': 'Robyn (Meta MMM)',
        'meridian': 'Meridian (Google Bayesian MMM)',
        'prophet': 'Prophet (Meta forecasting)',
        'neural_prophet': 'NeuralProphet',
        'orbit': 'Orbit (Uber Bayesian)',
        'causalml': 'CausalML (Uber)',
        'dowhy': 'DoWhy (Microsoft causal)',
        'econml': 'EconML (Microsoft)',
        'lifetimes': 'Lifetimes (CLV)',
        'kats': 'Kats (Facebook)',
        'mmm_stan': 'Stan MMM'
    }
    
    for repo_dir, repo_desc in mmm_repos.items():
        repo_path = sources_dir / repo_dir
        if repo_path.exists():
            logger.info(f"\n  Processing {repo_desc}...")
            examples = extract_production_examples(repo_path, repo_dir)
            all_examples.extend(examples)
            logger.info(f"    → {len(examples)} examples")
        else:
            logger.warning(f"    ✗ Not found: {repo_path}")
    
    logger.info(f"\n  Subtotal: {len(all_examples)} production examples\n")
    
    # =========================================================================
    # PART 2: Package API Examples (NumPyro, PyMC, JAX)
    # =========================================================================
    logger.info("PART 2: Extracting Package API Examples")
    logger.info("-" * 80)
    
    api_packages = {
        'numpyro': {
            'url': 'https://github.com/pyro-ppl/numpyro.git',
            'desc': 'NumPyro (Bayesian inference with JAX)'
        },
        'pymc': {
            'url': 'https://github.com/pymc-devs/pymc.git',
            'desc': 'PyMC (Bayesian modeling)'
        },
        'jax': {
            'url': 'https://github.com/google/jax.git',
            'desc': 'JAX (numerical computing)'
        },
    }
    
    api_dir = sources_dir / '_api_packages'
    api_dir.mkdir(exist_ok=True)
    
    for pkg_name, pkg_info in api_packages.items():
        logger.info(f"\n  {pkg_info['desc']}:")
        pkg_path = api_dir / pkg_name
        
        if clone_package_repo(pkg_name, pkg_info['url'], pkg_path):
            examples = extract_api_examples(pkg_path, pkg_name)
            all_examples.extend(examples)
            logger.info(f"    → {len(examples)} API examples")
    
    logger.info("")
    logger.info("="*80)
    logger.info(f"TOTAL EXAMPLES: {len(all_examples)}")
    logger.info("="*80)
    logger.info("")
    
    # Save combined dataset
    output_file = Path('fine_tuning/rag_production_plus_api.jsonl')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')
    
    logger.info(f"✓ Saved to {output_file}")
    logger.info("")
    
    # Analysis
    logger.info("Breakdown by source:")
    from collections import Counter
    repo_counts = Counter(ex['repo'] for ex in all_examples)
    for repo, count in repo_counts.most_common():
        logger.info(f"  {repo:30s}: {count:4d} examples")
    logger.info("")
    
    type_counts = Counter(ex['type'] for ex in all_examples)
    logger.info("By type:")
    for typ, count in type_counts.items():
        pct = (count / len(all_examples) * 100) if all_examples else 0
        logger.info(f"  {typ:20s}: {count:4d} ({pct:.1f}%)")
    logger.info("")
    
    # Content analysis
    logger.info("Content analysis:")
    keywords = {
        'Adstock/Saturation': sum(1 for ex in all_examples if any(k in ex['output'].lower() for k in ['adstock', 'saturation', 'hill'])),
        'PyMC/Bayesian': sum(1 for ex in all_examples if 'pymc' in ex['output'].lower() or 'pm.' in ex['output']),
        'JAX/NumPyro': sum(1 for ex in all_examples if 'jax' in ex['output'].lower() or 'numpyro' in ex['output'].lower()),
        'SVI/Inference': sum(1 for ex in all_examples if 'svi' in ex['output'].lower() or 'autoguide' in ex['output'].lower()),
        'Shape operations': sum(1 for ex in all_examples if any(k in ex['output'].lower() for k in ['reshape', 'broadcast', 'expand_dims'])),
        'Distributions': sum(1 for ex in all_examples if 'dist.' in ex['output'] or 'distribution' in ex['output'].lower()),
    }
    for kw, count in keywords.items():
        pct = (count / len(all_examples) * 100) if all_examples else 0
        logger.info(f"  {kw:20s}: {count:4d} ({pct:.1f}%)")
    logger.info("")
    
    logger.info("="*80)
    logger.info("✅ RAG DATASET READY!")
    logger.info("="*80)
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("1. Rebuild ChromaDB:")
    logger.info("   python build_rag_index.py --dataset rag_production_plus_api.jsonl")
    logger.info("")
    logger.info("2. Test with agent:")
    logger.info("   python cerebro_cli.py auto data.csv")
    logger.info("="*80)

if __name__ == '__main__':
    main()

