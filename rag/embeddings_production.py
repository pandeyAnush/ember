"""
Production-grade embeddings using BGE (BAAI General Embeddings)
Better semantic understanding than all-MiniLM-L6-v2
Dimension: 1024 (vs 384 in standard)

CRITICAL: BGE models require query instruction prefix for optimal performance.
Queries get: "Represent this sentence for searching relevant passages: " + query
"""

from sentence_transformers import SentenceTransformer
import numpy as np


class ProductionEmbeddingGenerator:
    """
    Uses BGE-large-en-v1.5 for superior embedding quality
    - Better semantic understanding
    - Larger embedding dimension (1024)
    - Production-proven model
    - Handles BGE query instruction prefix automatically
    """
    
    # BGE instruction for query embedding (model-specific, do not change)
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize with BGE model
        
        Args:
            model_name: HuggingFace model identifier
        """
        print(f"📦 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Embed a single text
        
        Args:
            text: Text to embed
            is_query: If True, prepends BGE query instruction (for search queries)
            
        Returns:
            Embedding vector
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        # Add query instruction if this is a search query
        text_to_embed = f"{self.QUERY_INSTRUCTION}{text}" if is_query else text
        
        embedding = self.model.encode(text_to_embed, convert_to_numpy=True)
        return embedding
    
    def embed_texts(self, texts: list, is_query: bool = False) -> np.ndarray:
        """
        Embed multiple texts (batch processing for efficiency)
        
        Args:
            texts: List of texts to embed
            is_query: If True, prepends BGE query instruction to all texts
            
        Returns:
            Array of embedding vectors
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # Add query instruction if needed
        texts_to_embed = [f"{self.QUERY_INSTRUCTION}{t}" if is_query else t for t in texts]
        
        print(f"Embedding {len(texts)} texts{'  (with query instruction)' if is_query else ''}...")
        embeddings = self.model.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=True)
        return embeddings
    
    # Alias methods for compatibility with existing Retriever class
    def embed_batch(self, texts: list, is_query: bool = False) -> np.ndarray:
        """Alias for embed_texts() - for compatibility with existing Retriever"""
        return self.embed_texts(texts, is_query=is_query)
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim
    
    # Alias for compatibility with existing Retriever class
    def get_dimension(self) -> int:
        """Alias for get_embedding_dimension() - for compatibility with existing Retriever"""
        return self.get_embedding_dimension()