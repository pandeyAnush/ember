from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from chunking import ChunkingStrategy
from document_loader import DocumentLoader
from typing import List, Dict
import numpy as np


class Retriever:
    """
    Orchestrates the entire retrieval pipeline:
    1. Takes documents
    2. Chunks them
    3. Embeds chunks
    4. Stores in vector DB
    5. Retrieves relevant chunks for queries
    """
    
    def __init__(self, chunking_strategy: ChunkingStrategy, embedding_model: str = "all-MiniLM-L6-v2"):
        self.chunking_strategy = chunking_strategy
        if embedding_model is None:
            # Production path: the caller supplies its own embedder (BGE) and
            # assigns .vector_store afterwards. Skip loading the built-in
            # MiniLM model entirely — it would just waste RAM and startup time.
            self.embedding_gen = None
            self.vector_store = None
        else:
            self.embedding_gen = EmbeddingGenerator(embedding_model)
            self.vector_store = VectorStore(self.embedding_gen.get_dimension())
    
    def index_documents(self, documents: List[Dict]):
        """
        Take raw documents and:
        1. Chunk them
        2. Generate embeddings
        3. Store in vector DB
        """
        print(f"Indexing {len(documents)} documents...")
        
        all_chunks = []
        all_embeddings = []
        
        for doc in documents:
            # Chunk the document
            chunks = self.chunking_strategy.chunk(doc['content'], doc['source'])
            all_chunks.extend(chunks)
            
            # Get embeddings for each chunk
            chunk_texts = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_gen.embed_batch(chunk_texts)
            all_embeddings.append(embeddings)
        
        # Combine all embeddings
        all_embeddings = np.vstack(all_embeddings)
        
        # Add to vector store
        self.vector_store.add(all_embeddings, all_chunks)
        
        print(f"Indexed {len(all_chunks)} chunks total")
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve the k most relevant chunks for a query.
        """
        # Generate embedding for the query
        query_embedding = self.embedding_gen.embed_text(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k=k)
        
        return results
    
    def save(self, save_dir: str):
        """Save the indexed data to disk."""
        self.vector_store.save(save_dir)
    
    def load(self, save_dir: str):
        """Load indexed data from disk."""
        self.vector_store.load(save_dir)


if __name__ == "__main__":
    from chunking import SentenceChunker
    
    # Test the full retrieval pipeline
    print("Setting up retriever...")
    retriever = Retriever(SentenceChunker(target_chunk_size=256))
    
    # Load sample documents
    loader = DocumentLoader()
    documents = loader.load_documents()
    
    if documents:
        print(f"\nIndexing {len(documents)} documents...")
        retriever.index_documents(documents)
        
        # Test retrieval
        test_queries = [
            "What is artificial intelligence?",
            "Tell me about machine learning",
            "How does natural language processing work?"
        ]
        
        print("\n" + "="*60)
        print("RETRIEVAL TEST")
        print("="*60)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = retriever.retrieve(query, k=2)
            
            for i, result in enumerate(results):
                print(f"\n  Result {i+1}:")
                print(f"    Source: {result['source']}")
                print(f"    Similarity: {result['similarity']:.4f}")
                print(f"    Content: {result['content'][:100]}...")
    else:
        print("No documents found. Run document_loader.py first to create sample data.")