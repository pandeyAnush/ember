from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
from datetime import datetime


class RetrieverEvaluator:
    """
    Evaluate how well the retriever is working.
    Did it find the right documents?
    """

    @staticmethod
    def precision_at_k(retrieved_chunks: List[Dict], relevant_sources: List[str], k: int = 5) -> float:
        """
        Precision@k: Of the top k retrieved documents, how many were relevant?
        E.g., if we retrieve 5 docs and 3 are relevant, precision@5 = 0.6
        """
        retrieved = retrieved_chunks[:k]
        num_relevant = sum(1 for chunk in retrieved if chunk['source'] in relevant_sources)
        return num_relevant / len(retrieved) if retrieved else 0.0

    @staticmethod
    def recall_at_k(retrieved_chunks: List[Dict], relevant_sources: List[str], k: int = 5) -> float:
        """
        Recall@k: Of all relevant documents, how many did we retrieve in top k?
        E.g., if there are 10 relevant docs and we got 3 in top 5, recall@5 = 0.3
        """
        retrieved = retrieved_chunks[:k]
        num_relevant_retrieved = sum(1 for chunk in retrieved if chunk['source'] in relevant_sources)
        num_relevant_total = len(relevant_sources)
        return num_relevant_retrieved / num_relevant_total if num_relevant_total > 0 else 0.0

    @staticmethod
    def mean_reciprocal_rank(retrieved_chunks: List[Dict], relevant_sources: List[str]) -> float:
        """
        MRR: How early did we find the first relevant document?
        If first relevant is at position 1: MRR = 1.0
        If first relevant is at position 3: MRR = 0.333
        """
        for i, chunk in enumerate(retrieved_chunks):
            if chunk['source'] in relevant_sources:
                return 1.0 / (i + 1)
        return 0.0


class GeneratorEvaluator:
    """
    Evaluate how good the LLM's responses are.
    This is harder without human labels, so we use heuristics.
    """

    @staticmethod
    def response_length(response: str) -> int:
        """Count words in response."""
        return len(response.split())

    @staticmethod
    def has_hallucination(response: str, context: str) -> bool:
        """
        Simple check: does response contain claims NOT in the context?
        This is a heuristic - not perfect but catches obvious hallucinations.
        """
        # Extract key entities/numbers from context
        context_words = set(re.findall(r'\b\w+\b', context.lower()))
        response_words = set(re.findall(r'\b\w+\b', response.lower()))

        # Words in response not in context (excluding common words)
        common_words = {'is', 'the', 'a', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        unusual_words = response_words - context_words - common_words

        # If there are many unusual words, might be hallucination
        # This is very crude - in real systems you'd want better detection
        return len(unusual_words) > len(context_words) * 0.3

    @staticmethod
    def response_relevance(response: str, question: str) -> float:
        """
        Simple heuristic: does the response address words from the question?
        E.g., if question asks about "Python" and response doesn't mention it, relevance is lower.
        """
        question_words = set(re.findall(r'\b\w{4,}\b', question.lower()))  # Words 4+ chars
        response_words = set(re.findall(r'\b\w{4,}\b', response.lower()))

        if not question_words:
            return 0.5

        overlap = len(question_words & response_words)
        return overlap / len(question_words)


class EvaluationLogger:
    """
    Log all evaluations to track system performance over time.
    This is crucial for LLMOps - you need metrics to know if things are improving.
    """

    def __init__(self, log_file: str = "evaluation_log.jsonl"):
        self.log_file = log_file

    def log_retrieval(self, query: str, retrieved_chunks: List[Dict], metrics: Dict):
        """Log a retrieval evaluation."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "retrieval",
            "query": query,
            "num_retrieved": len(retrieved_chunks),
            "metrics": metrics
        }
        self._write_log(entry)

    def log_generation(self, query: str, context: str, response: str, metrics: Dict):
        """Log a generation evaluation."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "generation",
            "query": query,
            "context_length": len(context),
            "response_length": len(response),
            "metrics": metrics
        }
        self._write_log(entry)

    def log_rag_round(self, query: str, retrieval_metrics: Dict, generation_metrics: Dict):
        """Log a full RAG round (retrieval + generation)."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "rag_round",
            "query": query,
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics
        }
        self._write_log(entry)

    def _write_log(self, entry: Dict):
        """Write entry to log file."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def get_summary(self) -> Dict:
        """Read log file and compute summary statistics."""
        entries = []
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        except FileNotFoundError:
            return {"error": "No log file found"}

        if not entries:
            return {"error": "No entries in log file"}

        # Aggregate metrics
        summary = {
            "total_queries": len(entries),
            "average_response_length": np.mean([e.get('response_length', 0) for e in entries]),
            "hallucination_rate": sum(1 for e in entries if e.get('metrics', {}).get('has_hallucination', False)) / len(entries)
        }

        return summary


if __name__ == "__main__":
    # Test evaluators
    print("Testing Evaluators")
    print("="*60)

    # Retrieval evaluation
    print("\n1. Retrieval Evaluation:")
    chunks = [
        {"source": "doc1", "content": "..."},
        {"source": "doc1", "content": "..."},
        {"source": "doc2", "content": "..."},
        {"source": "doc3", "content": "..."},
        {"source": "doc4", "content": "..."},
    ]
    relevant = ["doc1", "doc2"]

    precision = RetrieverEvaluator.precision_at_k(chunks, relevant, k=5)
    recall = RetrieverEvaluator.recall_at_k(chunks, relevant, k=5)
    mrr = RetrieverEvaluator.mean_reciprocal_rank(chunks, relevant)

    print(f"   Precision@5: {precision:.4f}")
    print(f"   Recall@5: {recall:.4f}")
    print(f"   MRR: {mrr:.4f}")

    # Generation evaluation
    print("\n2. Generation Evaluation:")
    context = "Python is a programming language. It's easy to learn."
    question = "What is Python?"
    response = "Python is a programming language known for its simplicity."

    is_hallucinating = GeneratorEvaluator.has_hallucination(response, context)
    relevance = GeneratorEvaluator.response_relevance(response, question)
    length = GeneratorEvaluator.response_length(response)

    print(f"   Hallucination: {is_hallucinating}")
    print(f"   Relevance: {relevance:.4f}")
    print(f"   Response length: {length} words")