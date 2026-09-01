from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import os


class EmbeddingGenerator:
    """Generate embeddings for text using Sentence Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a pre-trained model.
        all-MiniLM-L6-v2 is small (~80MB), fast, and pretty good quality.
        """
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def embed_text(self, text: str) -> np.ndarray:
        """Convert a single text string to an embedding vector."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Convert multiple texts to embeddings efficiently.
        Batching is much faster than encoding one-by-one.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings

    def get_dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self.model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    # Test the embedding generator
    generator = EmbeddingGenerator()

    test_texts = [
        "Machine learning is powerful",
        "Deep learning uses neural networks",
        "Natural language processing understands text"
    ]

    print("\nEmbedding single text:")
    single_embedding = generator.embed_text(test_texts[0])
    print(f"Shape: {single_embedding.shape}")
    print(f"First 5 values: {single_embedding[:5]}")

    print("\nEmbedding batch:")
    batch_embeddings = generator.embed_batch(test_texts)
    print(f"Shape: {batch_embeddings.shape}")
    print(f"Embedding 1 first 5 values: {batch_embeddings[0][:5]}")

    # Quick similarity test
    print("\nSimilarity check (cosine):")
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity([batch_embeddings[0]], batch_embeddings[1:])[0]
    for i, sim in enumerate(similarities):
        print(f"  Text 1 vs Text {i+2}: {sim:.4f}")