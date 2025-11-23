"""
RAG (Retrieval Augmented Generation) Backend for Cerebro.

Uses ChromaDB for vector storage and semantic retrieval of examples.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RAGBackend:
    """
    RAG system that retrieves relevant examples from a knowledge base
    to enhance LLM prompts with context.
    """
    
    def __init__(
        self,
        dataset_path: str = "fine_tuning/rag_production_only.jsonl",
        collection_name: str = "marketing_ds_examples",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = ".cerebro_rag"
    ):
        """
        Initialize RAG backend.
        
        Args:
            dataset_path: Path to JSONL file with examples
            collection_name: Name for ChromaDB collection
            embedding_model: SentenceTransformer model name
            persist_directory: Directory to persist ChromaDB
        """
        self.dataset_path = Path(dataset_path)
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        
        logger.info("Initializing RAG backend...")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
            logger.info(f"Loaded existing collection: {self.collection_name} "
                       f"({self.collection.count()} documents)")
        except Exception:
            logger.info(f"Creating new collection: {self.collection_name}")
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self._index_dataset()
    
    def _index_dataset(self):
        """Load and index the dataset into ChromaDB"""
        if not self.dataset_path.exists():
            logger.error(f"Dataset not found: {self.dataset_path}")
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        logger.info(f"Loading dataset from {self.dataset_path}...")
        
        # Load examples
        examples = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        
        logger.info(f"Loaded {len(examples)} examples")
        
        # Prepare data for indexing
        ids = []
        documents = []
        metadatas = []
        
        for i, ex in enumerate(examples):
            instruction = ex.get('instruction', '')
            output = ex.get('output', '')
            
            # Create searchable text (instruction + first 500 chars of output)
            searchable_text = f"{instruction}\n{output[:500]}"
            
            ids.append(f"ex_{i}")
            documents.append(searchable_text)
            metadatas.append({
                'instruction': instruction,
                'output': output[:4000],  # Limit output size
                'index': i
            })
        
        # Index in batches
        batch_size = 100
        logger.info(f"Indexing {len(examples)} examples in batches of {batch_size}...")
        
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(batch_docs).tolist()
            
            # Add to collection
            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_meta
            )
            
            if (i + batch_size) % 1000 == 0:
                logger.info(f"  Indexed {i + batch_size}/{len(examples)} examples")
        
        logger.info(f"✅ Indexed {len(examples)} examples")
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Retrieve most relevant examples for a query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_dict: Optional metadata filters
        
        Returns:
            List of examples with 'instruction' and 'output' keys
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )
        
        # Extract examples
        examples = []
        if results and results['metadatas']:
            for metadata in results['metadatas'][0]:
                examples.append({
                    'instruction': metadata.get('instruction', ''),
                    'output': metadata.get('output', '')
                })
        
        logger.debug(f"Retrieved {len(examples)} examples for query: {query[:100]}...")
        return examples
    
    def format_retrieved_examples(
        self,
        examples: List[Dict[str, str]],
        max_length: int = 3000
    ) -> str:
        """
        Format retrieved examples as a string for prompt augmentation.
        
        Args:
            examples: List of retrieved examples
            max_length: Maximum total length
        
        Returns:
            Formatted string with examples
        """
        if not examples:
            return ""
        
        formatted = "# Relevant Examples:\n\n"
        
        current_length = len(formatted)
        for i, ex in enumerate(examples, 1):
            example_text = f"## Example {i}:\n"
            example_text += f"Task: {ex['instruction']}\n\n"
            example_text += f"```python\n{ex['output'][:1000]}\n```\n\n"
            
            if current_length + len(example_text) > max_length:
                break
            
            formatted += example_text
            current_length += len(example_text)
        
        return formatted
    
    def augment_prompt(
        self,
        base_prompt: str,
        query: str,
        n_examples: int = 2,
        max_examples_length: int = 1200
    ) -> str:
        """
        Augment a base prompt with retrieved examples.
        
        Args:
            base_prompt: Original prompt
            query: Query for retrieval
            n_examples: Number of examples to retrieve (default 2 for speed)
            max_examples_length: Max length for examples section (default 1200)
        
        Returns:
            Augmented prompt
        """
        # Retrieve examples
        examples = self.retrieve(query, n_results=n_examples)
        
        if not examples:
            return base_prompt
        
        # Format examples
        examples_text = self.format_retrieved_examples(
            examples,
            max_length=max_examples_length
        )
        
        # Insert examples before the main task
        augmented = f"{examples_text}\n{'-'*80}\n\n{base_prompt}"
        
        return augmented
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return {
            'total_examples': self.collection.count(),
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model.__class__.__name__,
            'dataset_path': str(self.dataset_path),
            'persist_directory': str(self.persist_directory)
        }

