#!/usr/bin/env python3
"""
Rebuild ChromaDB RAG index with the enhanced dataset
"""

import shutil
import logging
from pathlib import Path
import sys

# Add cerebro to path
sys.path.insert(0, str(Path(__file__).parent))

from cerebro.llm.rag_backend import RAGBackend

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info("REBUILDING CHROMADB RAG INDEX")
    logger.info("="*80)
    logger.info("")
    
    # Remove old ChromaDB directory
    rag_dir = Path('.cerebro_rag')
    if rag_dir.exists():
        logger.info(f"Removing old RAG index: {rag_dir}")
        shutil.rmtree(rag_dir)
        logger.info("✓ Old index removed")
    else:
        logger.info("No existing index found")
    
    logger.info("")
    logger.info("Building new RAG index...")
    logger.info("")
    
    # Initialize RAG with the new enhanced dataset
    # This will automatically index the dataset
    rag = RAGBackend(
        dataset_path="fine_tuning/rag_production_plus_api.jsonl",
        collection_name="marketing_ds_examples",
        embedding_model="all-MiniLM-L6-v2",
        persist_directory=".cerebro_rag"
    )
    
    logger.info("")
    logger.info("="*80)
    logger.info(f"✅ RAG INDEX REBUILT: {rag.collection.count()} documents")
    logger.info("="*80)
    logger.info("")
    
    # Test retrieval
    logger.info("Testing retrieval with sample query...")
    query = "How to use NumPyro SVI with autoguide for Bayesian regression?"
    results = rag.augment_prompt(query, n_examples=3)
    
    logger.info(f"\nQuery: {query}")
    logger.info(f"Retrieved {len(results.split('---'))} examples")
    logger.info("")
    logger.info("Sample (first 500 chars):")
    logger.info(results[:500])
    logger.info("")
    logger.info("="*80)
    logger.info("✅ RAG SYSTEM READY!")
    logger.info("="*80)

if __name__ == '__main__':
    main()

