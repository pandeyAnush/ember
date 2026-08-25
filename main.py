"""
Main entry point for the RAG system.
This script demonstrates the complete workflow.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "rag"))

# Component imports
from document_loader import DocumentLoader
from chunking import SentenceChunker, FixedSizeChunker, ParagraphChunker
from retriever import Retriever
from llm_generator import OllamaGenerator
from rag_pipeline import RAGPipeline


def setup_documents():
    """Create sample documents for testing."""
    loader = DocumentLoader()
    
    sample_documents = {
        "sample.txt": """
        Artificial Intelligence is transforming industries worldwide.
        Machine learning models are becoming more sophisticated each year.
        Deep learning has revolutionized computer vision and speech recognition.
        
        Natural language processing enables computers to understand human text.
        Neural networks are the foundation of modern AI systems.
        Transformers have become the dominant architecture in NLP.
        
        Python is the most popular language for AI and machine learning.
        TensorFlow and PyTorch are leading frameworks for deep learning.
        Data preprocessing is often the most time-consuming part of ML projects.
        """,
        
        "rag_systems.txt": """
        Retrieval-Augmented Generation (RAG) combines retrieval and generation.
        RAG systems first retrieve relevant documents, then generate answers based on them.
        This approach reduces hallucinations compared to pure generative models.
        
        Vector databases like FAISS enable efficient similarity search.
        Embedding models convert text to numerical vectors.
        Chunking strategies affect retrieval quality significantly.
        
        RAG is useful for question-answering, summarization, and fact-checking.
        Popular RAG frameworks include LangChain and LlamaIndex.
        Fine-tuning both retrievers and generators improves RAG performance.
        """
    }
    
    for filename, content in sample_documents.items():
        path = Path("data") / filename
        path.parent.mkdir(exist_ok=True)
        path.write_text(content)
        print(f"✓ Created {filename}")
    
    return loader.load_documents()


def main():
    print("\n" + "="*70)
    print("RAG SYSTEM - Learning LLMOps in Practice")
    print("="*70)
    
    # Check Ollama availability
    print("\n[SETUP] Checking Ollama connection...")
    try:
        generator = OllamaGenerator(model="mistral")
    except Exception as e:
        print(f"✗ Ollama check failed. Make sure Ollama is running: ollama serve")
        print("  Install Ollama from ollama.ai and run: ollama pull mistral")
        return
    
    # Setup documents
    print("\n[SETUP] Preparing documents...")
    documents = setup_documents()
    
    if not documents:
        print("✗ No documents created")
        return
    
    print(f"✓ Loaded {len(documents)} documents")
    
    # Initialize RAG system with a chunking strategy
    print("\n[SETUP] Initializing RAG pipeline...")
    print("        Using: Sentence-based chunking + Semantic embeddings")
    
    chunker = SentenceChunker(target_chunk_size=256)
    retriever = Retriever(chunker)
    
    print("        Indexing documents...")
    retriever.index_documents(documents)
    
    # Create pipeline
    pipeline = RAGPipeline(retriever, generator, log_file="rag_results.jsonl")
    
    # Test queries
    print("\n" + "="*70)
    print("TESTING THE RAG SYSTEM")
    print("="*70)
    
    test_queries = [
        "What is artificial intelligence?",
        "How do RAG systems work?",
        "What programming language is best for AI?",
        "What are transformers in NLP?",
    ]
    
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}/{len(test_queries)}]")
        result = pipeline.query(query, num_context_chunks=2, llm_temperature=0.5)
        results.append(result)
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    total_time = sum(r['metrics']['total_time'] for r in results)
    avg_retrieval_quality = sum(
        r['metrics']['retrieval']['average_similarity'] for r in results
    ) / len(results)
    
    hallucination_count = sum(
        1 for r in results 
        if r['metrics']['generation']['has_hallucination']
    )
    
    avg_relevance = sum(
        r['metrics']['generation']['relevance'] for r in results
    ) / len(results)
    
    print(f"\nProcessed {len(results)} queries")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average retrieval quality (similarity): {avg_retrieval_quality:.4f}")
    print(f"Average response relevance: {avg_relevance:.4f}")
    print(f"Potential hallucinations: {hallucination_count}/{len(results)}")
    print(f"\nDetailed results saved to: rag_results.jsonl")
    
    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Type 'quit' to exit\n")
    
    while True:
        user_query = input("Ask a question: ").strip()
        if user_query.lower() == 'quit':
            break
        if not user_query:
            continue
        
        pipeline.query(user_query, num_context_chunks=3)


if __name__ == "__main__":
    main()