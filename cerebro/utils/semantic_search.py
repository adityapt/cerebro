"""
Semantic Search for Column Detection

Uses sentence embeddings to semantically match column names instead of keyword matching.
This is more robust and handles variations, languages, and unusual naming conventions.
"""

import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SemanticColumnMatcher:
    """
    Semantic column matcher using sentence embeddings.
    
    More robust than keyword matching:
    - Handles variations (weekid, wk_num, dt_week)
    - Language agnostic (periodo, fecha, semana)
    - Context-aware (revenue vs target vs sales)
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize semantic matcher.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self._model = None
        self._initialized = False
    
    def _initialize_model(self):
        """Lazy-load the model (only when needed)."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading semantic model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._initialized = True
            logger.info("✓ Semantic model loaded")
        except ImportError:
            logger.warning("sentence-transformers not installed. Install: pip install sentence-transformers")
            self._initialized = False
        except Exception as e:
            logger.warning(f"Failed to load semantic model: {e}")
            self._initialized = False
    
    def find_best_match(
        self,
        columns: List[str],
        query: str,
        threshold: float = 0.3
    ) -> Optional[str]:
        """
        Find the best matching column using semantic similarity.
        
        Args:
            columns: List of column names
            query: Semantic query (e.g., "date or time column")
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            Best matching column name, or None if no good match
        """
        self._initialize_model()
        
        if not self._initialized or not columns:
            return None
        
        try:
            # Encode query and columns
            query_embedding = self._model.encode([query])[0]
            column_embeddings = self._model.encode(columns)
            
            # Compute cosine similarities
            similarities = self._cosine_similarity(query_embedding, column_embeddings)
            
            # Find best match
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            
            logger.debug(f"Semantic search: '{query}' → '{columns[best_idx]}' (score: {best_score:.3f})")
            
            if best_score >= threshold:
                return columns[best_idx]
            else:
                logger.debug(f"No good match found (best score: {best_score:.3f} < threshold: {threshold})")
                return None
                
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return None
    
    def find_top_k_matches(
        self,
        columns: List[str],
        query: str,
        k: int = 3,
        threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Find top-k matching columns with scores.
        
        Args:
            columns: List of column names
            query: Semantic query
            k: Number of matches to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (column_name, score) tuples
        """
        self._initialize_model()
        
        if not self._initialized or not columns:
            return []
        
        try:
            # Encode
            query_embedding = self._model.encode([query])[0]
            column_embeddings = self._model.encode(columns)
            
            # Compute similarities
            similarities = self._cosine_similarity(query_embedding, column_embeddings)
            
            # Get top-k indices
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            
            # Build results
            results = []
            for idx in top_k_indices:
                score = float(similarities[idx])
                if score >= threshold:
                    results.append((columns[idx], score))
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between vector a and matrix b.
        
        Args:
            a: Query vector (1D)
            b: Matrix of vectors (2D)
            
        Returns:
            Array of similarity scores
        """
        # Normalize
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        
        # Dot product
        return np.dot(b_norm, a_norm)


# Singleton instance
_semantic_matcher = None


def get_semantic_matcher() -> SemanticColumnMatcher:
    """Get or create singleton semantic matcher."""
    global _semantic_matcher
    if _semantic_matcher is None:
        _semantic_matcher = SemanticColumnMatcher()
    return _semantic_matcher


def find_date_column_semantic(columns: List[str]) -> Optional[str]:
    """
    Find date column using semantic search.
    
    Args:
        columns: List of column names
        
    Returns:
        Best matching date column
    """
    matcher = get_semantic_matcher()
    return matcher.find_best_match(
        columns,
        query="date, time, or temporal column for time series analysis",
        threshold=0.25  # Lower threshold for date detection
    )


def find_kpi_column_semantic(columns: List[str], exclude: Optional[List[str]] = None) -> Optional[str]:
    """
    Find KPI/target column using semantic search.
    
    Args:
        columns: List of column names
        exclude: Columns to exclude (e.g., date, media channels)
        
    Returns:
        Best matching KPI column
    """
    if exclude:
        columns = [c for c in columns if c not in exclude]
    
    matcher = get_semantic_matcher()
    return matcher.find_best_match(
        columns,
        query="target, outcome, KPI, revenue, sales, conversions, or dependent variable for prediction",
        threshold=0.3
    )


def find_media_channels_semantic(
    columns: List[str],
    exclude: Optional[List[str]] = None
) -> List[str]:
    """
    Find media channel columns using semantic search.
    
    Args:
        columns: List of column names
        exclude: Columns to exclude (e.g., date, KPI)
        
    Returns:
        List of media channel columns
    """
    if exclude:
        columns = [c for c in columns if c not in exclude]
    
    matcher = get_semantic_matcher()
    matches = matcher.find_top_k_matches(
        columns,
        query="media channel, marketing spend, advertising, impressions, or marketing investment",
        k=len(columns),  # Get all matches
        threshold=0.25
    )
    
    # Return column names only
    return [col for col, score in matches]

