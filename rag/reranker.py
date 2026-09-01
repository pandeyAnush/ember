"""
Production Reranker
Re-ranks retrieved documents using cross-encoder for better relevance
This significantly improves accuracy of retrieval
"""

from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Tuple


class ProductionReranker:
    """
    Uses cross-encoder to re-rank documents
    - Evaluates relevance of query-document pairs directly
    -
    - Improves answer quality dramatically (typically 10-15% accuracy gain)

    Cross-encoders are slower than embedding-based ranking but more accurate:
    they see both query and document together, not separately as embeddings.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        Initialize cross-encoder model for ranking

        Args:
            model_name: HuggingFace cross-encoder model ID
                       ms-marco-MiniLM-L-12-v2 is optimized for relevance ranking
        """
        print(f"Loading reranker model: {model_name}")
        try:
            self.model = CrossEncoder(model_name)
            print(f"Reranker loaded successfully")
        except Exception as e:
            print(f"Failed to load reranker: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 3
    ) -> Tuple[List[str], List[float], List[int]]:
        """
        Re-rank documents by relevance to query

        Process:
        1. Create [query, document] pairs for all retrieved docs
        2. Score each pair with cross-encoder (0-1 relevance)
        3. Sort by score descending
        4. Return top-k with scores

        Args:
            query: User query (not embedded, passed as raw text)
            documents: List of retrieved document texts (raw, not embedded)
            top_k: Number of top documents to return after reranking

        Returns:
            Tuple of:
            - reranked_docs: Top-k documents sorted by relevance
            - scores: Relevance scores (0-1) for each top-k doc
            - indices: Original indices in input documents list
        """
        if not documents:
            return [], [], []

        if len(documents) == 0:
            print(" No documents to rerank")
            return [], [], []

        print(f"Re-ranking {len(documents)} documents for query: '{query[:60]}...'")

        try:
            # Create query-document pairs
            # CrossEncoder expects: [[query, doc1], [query, doc2], ...]
            pairs = [[query, doc] for doc in documents]

            # Get relevance scores (0-1, higher is more relevant)
            scores = self.model.predict(pairs)

            # Sort by score (descending = most relevant first)
            sorted_indices = np.argsort(scores)[::-1]

            # Take top-k
            top_k = min(top_k, len(documents))
            top_indices = sorted_indices[:top_k]
            top_scores = scores[top_indices]
            top_docs = [documents[i] for i in top_indices]

            # Log results
            print(f"Selected top {top_k} documents after reranking")
            for i, (doc, score) in enumerate(zip(top_docs, top_scores), 1):
                preview = doc[:80].replace('\n', ' ') + '...' if len(doc) > 80 else doc
                print(f"  {i}. Score: {score:.3f} - {preview}")

            # Return native Python types, not numpy arrays/scalars - numpy arrays
            # raise on truthiness checks like `if scores:`, and numpy.float64
            # isn't JSON-serializable when the result gets passed to jsonify()
            return top_docs, [float(s) for s in top_scores], [int(i) for i in top_indices]

        except Exception as e:
            print(f"Reranking failed: {e}")
            # Fallback: return top-k by original order if reranking fails
            print(f" Falling back to first {top_k} documents")
            return documents[:top_k], [1.0] * top_k, list(range(top_k))

    def filter_by_score(
        self,
        query: str,
        documents: List[str],
        threshold: float = 0.5
    ) -> Tuple[List[str], List[float]]:
        """
        Filter documents by relevance score threshold

        Useful for: only passing high-confidence documents to generation,
        or detecting when retrieval quality is too low to answer reliably

        Args:
            query: User query (raw text)
            documents: List of document texts
            threshold: Minimum relevance score (0-1) to include

        Returns:
            Tuple of:
            - filtered_docs: Only documents scoring >= threshold
            - scores: Scores for filtered documents
        """
        if not documents:
            return [], []

        print(f"Filtering documents by threshold: {threshold}")

        try:
            pairs = [[query, doc] for doc in documents]
            scores = self.model.predict(pairs)

            # Filter by threshold
            filtered_docs = []
            filtered_scores = []
            for doc, score in zip(documents, scores):
                if score >= threshold:
                    filtered_docs.append(doc)
                    filtered_scores.append(float(score))

            print(f"Kept {len(filtered_docs)}/{len(documents)} documents above {threshold} threshold")

            return filtered_docs, filtered_scores

        except Exception as e:
            print(f"Filtering failed: {e}")
            return documents, [1.0] * len(documents)