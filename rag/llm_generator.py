import requests
import json
import os
import time
from typing import List, Dict


class OllamaGenerator:
    """
    Generate responses using a local LLM via Ollama.

    Prerequisites:
    - Install Ollama from ollama.ai
    - Run: ollama pull mistral (or: ollama pull llama2)
    - Run: ollama serve (in another terminal)
    """

    def __init__(self, model: str = "mistral", base_url: str = None):
        self.model = model
        # OLLAMA_HOST lets docker-compose point this at the host's Ollama
        # (http://host.docker.internal:11434) instead of the container's own
        # localhost, which has no Ollama running on it
        self.base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.endpoint = f"{self.base_url}/api/generate"
        
        # Check if Ollama is running
        self._check_connection()
    
    def _check_connection(self):
        """Verify Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"✓ Connected to Ollama at {self.base_url}")
                self._print_available_models(response.json())
            else:
                print("⚠ Ollama server is not responding correctly")
        except requests.exceptions.ConnectionError:
            print(f"✗ Cannot connect to Ollama at {self.base_url}")
            print("  Make sure Ollama is running: ollama serve")
        except Exception as e:
            print(f"✗ Error checking Ollama: {e}")
    
    def _print_available_models(self, response: Dict):
        """Print available models from Ollama."""
        models = response.get('models', [])
        if models:
            print("  Available models:")
            for model in models:
                print(f"    - {model.get('name')}")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: The input prompt
            temperature: Controls randomness (0=deterministic, 1=creative)
            max_tokens: Maximum length of response
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "top_p": 0.9,
                "stream": False
            }
            
            response = requests.post(self.endpoint, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"Error: {response.status_code} - {response.text}"
        
        except requests.exceptions.Timeout:
            return "Error: Request timed out. The model might still be thinking."
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Make sure it's running."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_with_context(self, context: str, question: str, temperature: float = 0.5) -> str:
        """
        Generate a response given context and a question.
        This is the core RAG pattern.
        """
        prompt = f"""Based on the following context, answer the question. If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""
        
        return self.generate(prompt, temperature=temperature)


if __name__ == "__main__":
    # Test the generator
    generator = OllamaGenerator(model="mistral")
    
    print("\n" + "="*60)
    print("Testing LLM Generator")
    print("="*60)
    
    # Simple generation test
    print("\n1. Simple generation:")
    response = generator.generate("What is machine learning? Answer in one sentence.", temperature=0.5)
    print(f"   {response}")
    
    # Context-based generation (RAG pattern)
    print("\n2. Generation with context (RAG):")
    context = "Python is a programming language. It's known for being easy to read and write."
    question = "What is Python?"
    
    response = generator.generate_with_context(context, question)
    print(f"   {response}")