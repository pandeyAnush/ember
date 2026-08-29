"""
Production RAG Pipeline
Retrieval → Reranking → Generation → Evaluation
"""

from typing import Dict, Any
import time
import json
from datetime import datetime


class ProductionRAGPipeline:
    """
    Production-grade RAG pipeline with reranking
    
    Flow:
    1. Query → Embed with BGE query instruction
    2. Retrieve top-k documents from vector store
    3. Rerank for quality
    4. Generate answer with context
    5. Evaluate quality
    6. Log results
    """
    
    def __init__(self, retriever, embedding_generator, reranker, llm_generator, evaluation_logger=None):
        """
        Initialize RAG pipeline
        
        Args:
            retriever: Document retriever (with indexed vector_store)
            embedding_generator: ProductionEmbeddingGenerator instance
            reranker: Cross-encoder reranker
            llm_generator: LLM for generation
            evaluation_logger: Logger for metrics
        """
        self.retriever = retriever
        self.embedding_generator = embedding_generator
        self.reranker = reranker
        self.llm_generator = llm_generator
        self.evaluation_logger = evaluation_logger
        print("✅ Production RAG Pipeline initialized")
    
    def query(
        self,
        question: str,
        num_initial_retrieval: int = 10,
        num_final_chunks: int = 3
    ) -> Dict[str, Any]:
        """
        Execute full RAG pipeline
        
        Args:
            question: User question
            num_initial_retrieval: Initial retrieval count (before reranking)
            num_final_chunks: Final chunk count after reranking
            
        Returns:
            Dict with response, metrics, and source documents
        """
        pipeline_start = time.time()
        
        print(f"\n🚀 Processing query: {question}")
        print("=" * 60)
        
        # Step 1: Retrieve initial documents with BGE query instruction
        print(f"\n📚 Step 1: Initial Retrieval (top-{num_initial_retrieval})")
        retrieval_start = time.time()
        
        try:
            # Embed query with BGE instruction prefix for optimal retrieval
            query_embedding = self.embedding_generator.embed_text(question, is_query=True)
            retrieved_results = self.retriever.vector_store.search(query_embedding, k=num_initial_retrieval)
            
        except Exception as e:
            print(f"❌ Retrieval failed: {e}")
            return {
                "question": question,
                "response": f"Error during retrieval: {str(e)}",
                "status": "error"
            }
        
        retrieval_time = time.time() - retrieval_start
        retrieved_docs = [r["content"] for r in retrieved_results]
        
        print(f"✅ Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}s")
        
        # Calculate initial retrieval quality
        initial_similarity_scores = [r.get("similarity", 0) for r in retrieved_results]
        avg_initial_similarity = sum(initial_similarity_scores) / len(initial_similarity_scores) if initial_similarity_scores else 0
        
        # Step 2: Rerank documents
        print(f"\n🔄 Step 2: Reranking (filtering to top-{num_final_chunks})")
        rerank_start = time.time()
        
        try:
            reranked_docs, rerank_scores, rerank_indices = self.reranker.rerank(
                question,
                retrieved_docs,
                top_k=num_final_chunks
            )
        except Exception as e:
            print(f"⚠️  Reranking failed, using initial retrieval: {e}")
            reranked_docs = retrieved_docs[:num_final_chunks]
            rerank_scores = initial_similarity_scores[:num_final_chunks]
            rerank_indices = list(range(num_final_chunks))
        
        rerank_time = time.time() - rerank_start
        
        print(f"✅ Reranked to {len(reranked_docs)} documents in {rerank_time:.2f}s")
        
        # Step 3: Create context
        print(f"\n📄 Step 3: Context Assembly")
        context = "\n---\n".join([f"[Document {i+1}]\n{doc}" for i, doc in enumerate(reranked_docs)])
        
        # Step 4: Generate response
        print(f"\n✍️ Step 4: Response Generation")
        generation_start = time.time()
        
        generation_result = self.llm_generator.generate_with_context(
            question,
            context,
            num_context_chunks=len(reranked_docs)
        )
        
        generation_time = time.time() - generation_start
        response = generation_result.get("response", "")
        
        print(f"✅ Generated response in {generation_time:.2f}s")
        
        # Step 5: Evaluate
        print(f"\n📊 Step 5: Quality Evaluation")
        
        # Hallucination check
        hallucination_result = self.llm_generator.check_hallucination(response, context)
        has_hallucination = hallucination_result.get("has_hallucination", False)
        hallucination_confidence = hallucination_result.get("confidence", 0)
        
        # Relevance - semantic similarity (BGE) between the question and the
        # answer. This reflects on-topic-ness far better than word overlap, which
        # unfairly penalises short, direct answers (a concise correct answer
        # shares few literal words with the question). Reuses the query embedding
        # already computed above.
        try:
            import numpy as np
            resp_embedding = self.embedding_generator.embed_text(response, is_query=False)
            denom = (np.linalg.norm(query_embedding) * np.linalg.norm(resp_embedding)) + 1e-9
            relevance = float(np.dot(query_embedding, resp_embedding) / denom)
            relevance = max(0.0, min(1.0, relevance))
        except Exception:
            relevance = self.llm_generator.calculate_relevance(response, question)
        
        # Reranking quality
        avg_rerank_score = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0
        
        pipeline_time = time.time() - pipeline_start
        
        print(f"⏱️  Total pipeline time: {pipeline_time:.2f}s")
        
        # Map rerank indices back to original source documents
        # rerank_indices tells us which of the initially retrieved docs made it to top-k
        source_docs = []
        for idx in rerank_indices:
            if idx < len(retrieved_results):
                source_docs.append(retrieved_results[idx].get("source", "Unknown"))
            else:
                source_docs.append("Unknown")
        
        # Build the final reranked chunks as OBJECTS (content + source + similarity)
        # so the frontend can show source and match% - it does chunk.content /
        # chunk.source / chunk.similarity, which breaks on bare strings.
        retrieved_chunks = []
        for rank, idx in enumerate(rerank_indices):
            if idx < len(retrieved_results):
                r = retrieved_results[idx]
                retrieved_chunks.append({
                    "content": r.get("content", ""),
                    "source": r.get("source", "Unknown"),
                    # original vector similarity (0-1) reads better than the raw
                    # cross-encoder score for a "% match" display
                    "similarity": float(r.get("similarity", 0)),
                })

        # Build result. Metric field names match what the frontend reads
        # (num_retrieved / average_similarity / retrieval_time / generation_time / total_time).
        result = {
            "question": question,
            "response": response,
            "retrieved_chunks": retrieved_chunks,
            "source_documents": source_docs,
            "metrics": {
                "retrieval": {
                    "num_retrieved": len(retrieved_docs),
                    "average_similarity": float(avg_initial_similarity),
                    "retrieval_time": retrieval_time,
                    "final_count": len(reranked_docs),
                    "avg_rerank_score": float(avg_rerank_score),
                },
                "generation": {
                    "generation_time": generation_time,
                    "response_length": len(response),
                    "has_hallucination": has_hallucination,
                    "hallucination_confidence": hallucination_confidence,
                    "relevance": float(relevance),
                },
                "total_time": pipeline_time,
            },
            "status": "success"
        }
        
        # Log results if logger is available
        if self.evaluation_logger:
            try:
                # EvaluationLogger likely has log_result() method
                # If not, handle gracefully
                if hasattr(self.evaluation_logger, 'log_result'):
                    self.evaluation_logger.log_result(result)
                else:
                    print(f"⚠️  EvaluationLogger missing log_result method")
            except Exception as e:
                print(f"⚠️  Logging failed: {e}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("📈 QUALITY SUMMARY:")
        print(f"  Retrieval Quality: {avg_rerank_score:.1%} (after reranking)")
        print(f"  Response Relevance: {relevance:.1%}")
        print(f"  Hallucination Risk: {'🚨 DETECTED' if has_hallucination else '✅ OK'}")
        print(f"  Total Time: {pipeline_time:.2f}s")
        print("=" * 60)
        
        return result


    def query_stream(self, question, num_initial_retrieval=10, num_final_chunks=3):
        """Streaming variant of query(). A generator that yields event dicts:
          {"type": "context", ...}  once, with retrieved chunks + retrieval metrics
          {"type": "token", "text": ...}  many times, as the answer is generated
          {"type": "done", ...}  once, with the full response and final metrics
        Lets the UI show sources immediately and stream the answer token-by-token."""
        import numpy as np
        pipeline_start = time.time()

        # 1. Retrieve
        retrieval_start = time.time()
        query_embedding = self.embedding_generator.embed_text(question, is_query=True)
        retrieved_results = self.retriever.vector_store.search(query_embedding, k=num_initial_retrieval)
        retrieval_time = time.time() - retrieval_start
        retrieved_docs = [r["content"] for r in retrieved_results]
        initial_scores = [r.get("similarity", 0) for r in retrieved_results]
        avg_initial_similarity = sum(initial_scores) / len(initial_scores) if initial_scores else 0

        # 2. Rerank
        try:
            reranked_docs, rerank_scores, rerank_indices = self.reranker.rerank(
                question, retrieved_docs, top_k=num_final_chunks)
        except Exception:
            reranked_docs = retrieved_docs[:num_final_chunks]
            rerank_indices = list(range(min(num_final_chunks, len(retrieved_results))))

        retrieved_chunks = []
        for idx in rerank_indices:
            if idx < len(retrieved_results):
                r = retrieved_results[idx]
                retrieved_chunks.append({
                    "content": r.get("content", ""),
                    "source": r.get("source", "Unknown"),
                    "similarity": float(r.get("similarity", 0)),
                })

        # Emit context + retrieval metrics up front (UI shows sources immediately)
        yield {
            "type": "context",
            "retrieved_chunks": retrieved_chunks,
            "metrics": {"retrieval": {
                "num_retrieved": len(retrieved_docs),
                "average_similarity": float(avg_initial_similarity),
                "retrieval_time": retrieval_time,
            }},
        }

        # 3. Stream generation
        context = "\n---\n".join([f"[Document {i+1}]\n{doc}" for i, doc in enumerate(reranked_docs)])
        generation_start = time.time()
        full = ""
        try:
            for piece in self.llm_generator.generate_with_context_stream(question, context):
                full += piece
                yield {"type": "token", "text": piece}
        except Exception as e:
            err = f"ERROR during generation: {e}"
            full = full or err
            yield {"type": "token", "text": err}
        generation_time = time.time() - generation_start

        # 4. Final metrics
        hallucination = self.llm_generator.check_hallucination(full, context)
        try:
            resp_embedding = self.embedding_generator.embed_text(full, is_query=False)
            denom = (np.linalg.norm(query_embedding) * np.linalg.norm(resp_embedding)) + 1e-9
            relevance = max(0.0, min(1.0, float(np.dot(query_embedding, resp_embedding) / denom)))
        except Exception:
            relevance = self.llm_generator.calculate_relevance(full, question)

        result = {
            "question": question,
            "response": full,
            "retrieved_chunks": retrieved_chunks,
            "metrics": {
                "retrieval": {
                    "num_retrieved": len(retrieved_docs),
                    "average_similarity": float(avg_initial_similarity),
                    "retrieval_time": retrieval_time,
                },
                "generation": {
                    "generation_time": generation_time,
                    "response_length": len(full),
                    "has_hallucination": hallucination.get("has_hallucination", False),
                    "relevance": float(relevance),
                },
                "total_time": time.time() - pipeline_start,
            },
            "status": "success",
        }
        if self.evaluation_logger and hasattr(self.evaluation_logger, "log_result"):
            try:
                self.evaluation_logger.log_result(result)
            except Exception:
                pass

        yield {"type": "done", **result}


if __name__ == "__main__":
    print("Production RAG Pipeline Module")
    print("Use with retriever, embedding_generator, reranker, and llm_generator")