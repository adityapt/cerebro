"""
Rebuild RAG with ONLY production MMM code - no synthetic examples
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_production_examples(repo_path, repo_name):
    """Extract ALL relevant code from production MMM repos"""
    examples = []
    
    # File patterns to include - be aggressive
    include_patterns = [
        '**/*.py'  # Get ALL Python files, filter by content
    ]
    
    # Skip patterns
    skip_patterns = [
        'test_', '_test.py', 'conftest', '__init__.py', 'setup.py',
        '__pycache__', '.pyc', 'example.py', 'demo.py', 'tutorial',
        'docs/', 'doc/', 'notebooks/', 'examples/'
    ]
    
    for pattern in include_patterns:
        for file_path in repo_path.rglob(pattern):
            # Check skip patterns
            if any(skip in str(file_path) for skip in skip_patterns):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Skip tiny files only (we'll chunk large ones)
                if len(content) < 200:
                    continue
                
                # Chunk large files into multiple examples
                if len(content) > 10000:
                    # Split into ~5000 char chunks
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
                    
                    # Create example for each chunk
                    for i, chunk in enumerate(chunks):
                        if len(chunk) > 500:  # Skip tiny chunks
                            rel_path = file_path.relative_to(repo_path)
                            examples.append({
                                'instruction': f'Production MMM code from {repo_name}: {rel_path} (part {i+1}/{len(chunks)})',
                                'output': chunk,
                                'repo': repo_name,
                                'file': str(rel_path),
                                'type': 'production_code'
                            })
                    
                    logger.info(f"✓ Extracted {repo_name}/{rel_path} ({len(content)} chars → {len(chunks)} chunks)")
                    continue
                
                # Must have MMM/ML-relevant code (broader filter)
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
                # Need at least 2 keywords to avoid false positives
                keyword_count = sum(1 for kw in relevant_keywords if kw in content_lower)
                if keyword_count < 2:
                    continue
                
                # Create example
                rel_path = file_path.relative_to(repo_path)
                examples.append({
                    'instruction': f'Production MMM code from {repo_name}: {rel_path}',
                    'output': content,
                    'repo': repo_name,
                    'file': str(rel_path),
                    'type': 'production_code'
                })
                
                logger.info(f"✓ Extracted {repo_name}/{rel_path} ({len(content)} chars)")
                
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                continue
    
    return examples

def main():
    logger.info("="*80)
    logger.info("REBUILDING RAG WITH PRODUCTION CODE ONLY")
    logger.info("="*80)
    logger.info("")
    
    # Source directory
    sources_dir = Path('fine_tuning/mmm_sources')
    
    if not sources_dir.exists():
        logger.error(f"Sources directory not found: {sources_dir}")
        return
    
    # Extract from all repos
    all_examples = []
    
    repos = {
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
    
    for repo_dir, repo_desc in repos.items():
        repo_path = sources_dir / repo_dir
        if repo_path.exists():
            logger.info(f"\nProcessing {repo_desc}...")
            examples = extract_production_examples(repo_path, repo_dir)
            all_examples.extend(examples)
            logger.info(f"  → Extracted {len(examples)} files")
        else:
            logger.warning(f"  ✗ Not found: {repo_path}")
    
    logger.info("")
    logger.info("="*80)
    logger.info(f"TOTAL PRODUCTION EXAMPLES: {len(all_examples)}")
    logger.info("="*80)
    logger.info("")
    
    # Save production-only dataset
    output_file = Path('fine_tuning/rag_production_only.jsonl')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')
    
    logger.info(f"✓ Saved to {output_file}")
    logger.info("")
    
    # Show breakdown by repo
    logger.info("Breakdown by repository:")
    from collections import Counter
    repo_counts = Counter(ex['repo'] for ex in all_examples)
    for repo, count in repo_counts.most_common():
        logger.info(f"  {repo:30s}: {count:4d} examples")
    logger.info("")
    
    # Show content types
    logger.info("Content analysis:")
    keywords = {
        'Adstock': sum(1 for ex in all_examples if 'adstock' in ex['output'].lower()),
        'Saturation': sum(1 for ex in all_examples if 'saturation' in ex['output'].lower() or 'hill' in ex['output'].lower()),
        'PyMC/Bayesian': sum(1 for ex in all_examples if 'pymc' in ex['output'].lower() or 'pm.' in ex['output']),
        'JAX/NumPyro': sum(1 for ex in all_examples if 'jax' in ex['output'].lower() or 'numpyro' in ex['output'].lower()),
        'Likelihood': sum(1 for ex in all_examples if 'likelihood' in ex['output'].lower()),
        'Priors': sum(1 for ex in all_examples if 'prior' in ex['output'].lower()),
    }
    for kw, count in keywords.items():
        pct = (count / len(all_examples) * 100) if all_examples else 0
        logger.info(f"  {kw:20s}: {count:4d} examples ({pct:.1f}%)")
    logger.info("")
    
    logger.info("="*80)
    logger.info("NEXT STEP: Rebuild ChromaDB with production code only")
    logger.info("Run: python build_rag_index.py --dataset rag_production_only.jsonl")
    logger.info("="*80)

if __name__ == '__main__':
    main()

