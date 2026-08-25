import faiss
import numpy as np
from typing import List, Dict, Tuple
import json
from pathlib import Path


class VectorStore:
    """
    Stores embeddings using FAISS and keeps metadata about chunks.
    FAISS makes similarity search super fast even with millions of vectors.
    """
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        # Create a simple L2 distance index (Euclidean distance)
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks = []  # Store the actual chunk data
    
    def add(self, embeddings: np.ndarray, chunks: List[Dict]):
        """Add embeddings and their corresponding chunk metadata."""
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        # FAISS expects float32
        embeddings = embeddings.astype(np.float32)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store chunk metadata
        self.chunks.extend(chunks)
        
        print(f"Added {len(chunks)} chunks. Total chunks: {len(self.chunks)}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Find the k most similar chunks to a query embedding.
        Returns chunks with similarity score.
        """
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # FAISS returns distances and indices
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:  # Valid result
                chunk = self.chunks[idx].copy()
                # Convert L2 distance to similarity (smaller distance = higher similarity)
                # Normalize to 0-1 range for readability
                chunk["distance"] = float(distances[0][i])
                chunk["similarity"] = 1 / (1 + chunk["distance"])  # Convert distance to similarity
                results.append(chunk)
        
        return results
    
    def save(self, save_dir: str):
        """Save the index and chunks to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "faiss.index"))
        
        # Save chunks as JSON
        with open(save_path / "chunks.json", "w") as f:
            json.dump(self.chunks, f)
        
        print(f"Vector store saved to {save_dir}")
    
    def load(self, save_dir: str):
        """Load the index and chunks from disk."""
        save_path = Path(save_dir)
        
        # Load FAISS index
        self.index = faiss.read_index(str(save_path / "faiss.index"))
        
        # Load chunks
        with open(save_path / "chunks.json", "r") as f:
            self.chunks = json.load(f)
        
        print(f"Vector store loaded from {save_dir}. Total chunks: {len(self.chunks)}")
    
    def __len__(self):
        return len(self.chunks)


if __name__ == "__main__":
    # Test the vector store
    embedding_dim = 384
    store = VectorStore(embedding_dim)
    
    # Create dummy embeddings and chunks
    dummy_embeddings = np.random.rand(3, embedding_dim)
    dummy_chunks = [
        {"content": "Text about AI", "source": "doc1", "chunk_id": 0},
        {"content": "Text about ML", "source": "doc1", "chunk_id": 1},
        {"content": "Text about NLP", "source": "doc2", "chunk_id": 0}
    ]
    
    print("Adding chunks to store...")
    store.add(dummy_embeddings, dummy_chunks)
    
    print("\nSearching for similar chunks...")
    query_embedding = np.random.rand(embedding_dim)
    results = store.search(query_embedding, k=2)
    
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['content'][:30]}... (similarity: {result['similarity']:.4f})")