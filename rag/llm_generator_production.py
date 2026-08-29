"""
Production LLM Generator using Ollama
- Llama 3.1 8B (hardware-appropriate for 16GB M4 MacBook Air)
- Better quality responses than Mistral
- Structured output with metrics
- Reduced hallucinations via prompt engineering
"""

import os
import requests
import json
from typing import Dict, Any


class ProductionLLMGenerator:
    """
    Generates responses using Ollama LLM
    - Hardware-appropriate model sizing
    - Direct API calls (no double-calling)
    - Quality metrics tracking
    """
    
    def __init__(
        self,
        model: str = "llama3.1:8b",  # Hardware-appropriate for 16GB RAM
        base_url: str = None,
        temperature: float = 0.3
    ):
        # OLLAMA_HOST env override (e.g. http://host.docker.internal:11434 in a
        # container); defaults to the local Ollama server.
        base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        """
        Initialize LLM generator
        
        Args:
            model: Ollama model name
                   llama3.1:8b - RECOMMENDED for 16GB M4 MacBook Air (balanced quality/speed)
                   mistral - faster but lower quality
                   (llama2:70b requires 40GB+ RAM - not recommended for this hardware)
            base_url: Ollama server URL
            temperature: 0.1-0.3 for factual, 0.5+ for creative
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        print(f"⚙️ Initializing LLM: {model}")
        print(f"⚙️  Temperature: {temperature} (lower = more focused/factual)")
        print(f"📍 Ollama URL: {base_url}")
    
    def generate_with_context(
        self,
        question: str,
        context: str,
        num_context_chunks: int = 3
    ) -> Dict[str, Any]:
        """
        Generate response using context (RAG)
        Single API call - no double-generation overhead
        
        Args:
            question: User question
            context: Retrieved context from documents
            num_context_chunks: Number of chunks used (for logging)
            
        Returns:
            Dict with response and metadata
        """
        # Prompt tuned for DIRECT, confident answers. The old prompt over-warned
        # about "NOT FOUND IN CONTEXT" and "cite which part", which made Llama 8B
        # hedge, second-guess itself, and narrate its reasoning instead of just
        # answering - even when the answer was clearly in the context.
        system_prompt = """You answer questions about the user's documents using ONLY the provided context.

Rules:
- Answer immediately and confidently. The first sentence must be the answer itself.
- Use ONLY the context. NEVER add tools, facts, names, or examples from your own knowledge or from "typical" cases.
- Do NOT hedge or second-guess. Do not discuss whether the context is "explicit" or "sufficient" - if the context states something, report it plainly as fact.
- Do NOT narrate reasoning or refer to "Document 1/2/3", "the context", or "the pipeline mentioned". Just answer.
- For broad or "tell me about / summarise / what is this about" questions, ALWAYS give your best short overview synthesised from whatever the context contains. Never refuse a summary.
- Be concise: a short paragraph or a tight bullet list, no preamble.
- Reply "The document doesn't cover that." ONLY when the question is about a completely different topic with nothing related in the context. When in doubt, answer."""

        user_prompt = f"""Context:
{context}

Question: {question}

Answer concisely, using the context above."""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": user_prompt,
                    "system": system_prompt,
                    "stream": False,
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "top_k": 40,
                },
                timeout=120
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama error: {response.text}")
            
            result = response.json()
            generated_text = result.get("response", "").strip()
            
            return {
                "response": generated_text,
                "model": self.model,
                "context_chunks": num_context_chunks,
                "status": "success"
            }
            
        except requests.exceptions.ConnectionError:
            return {
                "response": "ERROR: Cannot connect to Ollama. Ensure Ollama is running: ollama serve",
                "status": "error"
            }
        except Exception as e:
            return {
                "response": f"ERROR: {str(e)}",
                "status": "error"
            }
    
    def generate_with_context_stream(self, question: str, context: str):
        """Same as generate_with_context but STREAMS the answer token-by-token.
        Yields text pieces as they are produced by Ollama. Uses the identical
        system/user prompt so streamed answers match the non-streamed ones."""
        system_prompt = """You answer questions about the user's documents using ONLY the provided context.

Rules:
- Answer immediately and confidently. The first sentence must be the answer itself.
- Use ONLY the context. NEVER add tools, facts, names, or examples from your own knowledge or from "typical" cases.
- Do NOT hedge or second-guess. Do not discuss whether the context is "explicit" or "sufficient" - if the context states something, report it plainly as fact.
- Do NOT narrate reasoning or refer to "Document 1/2/3", "the context", or "the pipeline mentioned". Just answer.
- For broad or "tell me about / summarise / what is this about" questions, ALWAYS give your best short overview synthesised from whatever the context contains. Never refuse a summary.
- Be concise: a short paragraph or a tight bullet list, no preamble.
- Reply "The document doesn't cover that." ONLY when the question is about a completely different topic with nothing related in the context. When in doubt, answer."""

        user_prompt = f"""Context:
{context}

Question: {question}

Answer concisely, using the context above."""

        with requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": True,
                "temperature": self.temperature,
                "top_p": 0.9,
                "top_k": 40,
            },
            stream=True,
            timeout=180,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                piece = data.get("response", "")
                if piece:
                    yield piece
                if data.get("done"):
                    break

    def check_hallucination(self, response: str, context: str) -> Dict[str, Any]:
        """
        Check if response contains potential hallucinations
        Uses heuristic analysis (word overlap, context grounding)
        
        Args:
            response: Generated response
            context: Original context
            
        Returns:
            Hallucination analysis (heuristic-based, not LLM-driven to avoid double calls)
        """
        # Heuristic-based hallucination detection (faster, no extra LLM call)
        response_lower = response.lower()
        context_lower = context.lower()
        
        # Red flags
        has_uncertain_language = any(phrase in response_lower for phrase in [
            "i think", "probably", "might", "could be", "unclear", "ambiguous"
        ])
        
        has_not_found = "not found" in response_lower or "unable to" in response_lower
        
        # Check if response content appears in context
        response_words = set(response_lower.split())
        context_words = set(context_lower.split())
        overlap = len(response_words & context_words) / max(len(response_words), 1) if response_words else 0
        
        # Hallucination likely if low overlap AND uncertain language AND not explicitly saying "not found"
        has_hallucination = (overlap < 0.3 and has_uncertain_language and not has_not_found)
        confidence = min(abs(overlap - 0.5), 1.0)  # Confidence peaks at extremes
        
        return {
            "has_hallucination": has_hallucination,
            "confidence": confidence,
            "explanation": f"Word overlap: {overlap:.1%}" + (
                " (uncertain language detected)" if has_uncertain_language else ""
            )
        }
    
    def calculate_relevance(self, response: str, question: str) -> float:
        """
        Calculate relevance score (0-1)
        Uses word overlap and semantic heuristics
        
        Args:
            response: Generated response
            question: Original question
            
        Returns:
            Relevance score 0-1
        """
        # Word overlap heuristic
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        
        if not question_words:
            return 0.0
        
        overlap = len(question_words & response_words)
        relevance = min(overlap / max(len(question_words), 1), 1.0)
        
        # Penalty for responses saying "not found"
        if "not found" in response.lower() or "unable" in response.lower():
            relevance *= 0.8
        
        # Bonus for actually attempting an answer
        if len(response) > 50:  # Substantial response
            relevance = min(relevance + 0.1, 1.0)
        
        return relevance