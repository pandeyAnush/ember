from retriever import Retriever
from llm_generator import OllamaGenerator
from evaluation import RetrieverEvaluator, GeneratorEvaluator, EvaluationLogger
from chunking import ChunkingStrategy
from typing import List, Dict
import time


class RAGPipeline:
    """
    Complete RAG pipeline: retrieval + generation + evaluation.
    This is what you actually use to ask questions.
    """
    
    def __init__(self, retriever: Retriever, generator: OllamaGenerator, 
                 log_file: str = "evaluation_log.jsonl"):
        self.retriever = retriever
        self.generator = generator
        self.logger = EvaluationLogger(log_file)
    
    def query(self, question: str, num_context_chunks: int = 3, 
              llm_temperature: float = 0.5, log: bool = True) -> Dict:
        """
        Run a complete RAG query:
        1. Retrieve relevant documents
        2. Generate answer based on documents
        3. Evaluate both retrieval and generation
        4. Log everything
        """
        
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print(f"{'='*60}")
        
        # Step 1: Retrieve
        print("\n[1/3] Retrieving context...")
        retrieved = self.retriever.retrieve(question, k=num_context_chunks)
        retrieval_time = time.time() - start_time
        
        print(f"      Found {len(retrieved)} relevant chunks in {retrieval_time:.2f}s")
        
        # Show what we retrieved
        for i, chunk in enumerate(retrieved):
            print(f"      {i+1}. {chunk['source']} (similarity: {chunk['similarity']:.4f})")
        
        # Step 2: Build context and generate
        print("\n[2/3] Generating response...")
        context = "\n\n".join([chunk['content'] for chunk in retrieved])
        
        generation_start = time.time()
        response = self.generator.generate_with_context(context, question, temperature=llm_temperature)
        generation_time = time.time() - generation_start
        
        print(f"      Generated response in {generation_time:.2f}s")
        print(f"\n      Answer: {response}\n")
        
        # Step 3: Evaluate
        print("[3/3] Evaluating...")
        
        retrieval_metrics = {
            "num_retrieved": len(retrieved),
            "average_similarity": sum(c['similarity'] for c in retrieved) / len(retrieved) if retrieved else 0,
            "retrieval_time": retrieval_time
        }
        
        generation_metrics = {
            "response_length": GeneratorEvaluator.response_length(response),
            "has_hallucination": GeneratorEvaluator.has_hallucination(response, context),
            "relevance": GeneratorEvaluator.response_relevance(response, question),
            "generation_time": generation_time
        }
        
        print(f"      Retrieval quality: {retrieval_metrics['average_similarity']:.4f} avg similarity")
        print(f"      Response relevance: {generation_metrics['relevance']:.4f}")
        print(f"      Potential hallucination: {generation_metrics['has_hallucination']}")
        
        # Log it
        if log:
            self.logger.log_rag_round(question, retrieval_metrics, generation_metrics)
        
        # Package results
        result = {
            "question": question,
            "retrieved_chunks": retrieved,
            "context": context,
            "response": response,
            "metrics": {
                "retrieval": retrieval_metrics,
                "generation": generation_metrics,
                "total_time": time.time() - start_time
            }
        }
        
        return result
    
    def batch_query(self, questions: List[str]) -> List[Dict]:
        """Run multiple queries and collect results."""
        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)
        return results
    
    def get_evaluation_summary(self) -> Dict:
        """Get summary of all evaluations so far."""
        return self.logger.get_summary()


if __name__ == "__main__":
    print("RAG Pipeline is a module. Use it from main.py")