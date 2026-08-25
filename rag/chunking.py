from typing import List, Dict
import re


class ChunkingStrategy:
    """Base class for different chunking approaches."""
    
    def chunk(self, text: str, source: str) -> List[Dict]:
        """
        Splits text into chunks. Returns list of dicts with:
        - content: the chunk text
        - source: where it came from
        - chunk_id: which chunk number
        """
        raise NotImplementedError


class FixedSizeChunker(ChunkingStrategy):
    """Simple: split text into fixed-size chunks with overlap."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, source: str) -> List[Dict]:
        chunks = []
        # Split by character, maintaining overlap
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i : i + self.chunk_size]
            if chunk_text.strip():  # Skip empty chunks
                chunks.append({
                    "content": chunk_text,
                    "source": source,
                    "chunk_id": len(chunks),
                    "strategy": "fixed_size"
                })
        return chunks


class SentenceChunker(ChunkingStrategy):
    """Smarter: split by sentences, then group into target size."""
    
    def __init__(self, target_chunk_size: int = 512):
        self.target_chunk_size = target_chunk_size
    
    def chunk(self, text: str, source: str) -> List[Dict]:
        # Split by sentence (simple regex - not perfect but works)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            # If adding this sentence doesn't exceed target, add it
            if len(current_chunk) + len(sentence) < self.target_chunk_size:
                current_chunk += " " + sentence
            else:
                # Save current chunk and start a new one
                if current_chunk.strip():
                    chunks.append({
                        "content": current_chunk.strip(),
                        "source": source,
                        "chunk_id": chunk_id,
                        "strategy": "sentence"
                    })
                    chunk_id += 1
                current_chunk = sentence
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "source": source,
                "chunk_id": chunk_id,
                "strategy": "sentence"
            })
        
        return chunks


class ParagraphChunker(ChunkingStrategy):
    """Semantic-ish: split by paragraphs first, then combine if needed."""
    
    def __init__(self, target_chunk_size: int = 512):
        self.target_chunk_size = target_chunk_size
    
    def chunk(self, text: str, source: str) -> List[Dict]:
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            # Try to fit paragraph into current chunk
            if len(current_chunk) + len(para) < self.target_chunk_size:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                # Current chunk is full, save it
                if current_chunk.strip():
                    chunks.append({
                        "content": current_chunk.strip(),
                        "source": source,
                        "chunk_id": chunk_id,
                        "strategy": "paragraph"
                    })
                    chunk_id += 1
                current_chunk = para
        
        # Save the last chunk
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "source": source,
                "chunk_id": chunk_id,
                "strategy": "paragraph"
            })
        
        return chunks


if __name__ == "__main__":
    # Test all chunking strategies
    sample_text = """
    Artificial Intelligence is transforming industries worldwide. Machine learning models are becoming more sophisticated.
    
    Natural language processing enables computers to understand human text. Deep learning has revolutionized computer vision.
    
    The field moves quickly. New techniques emerge constantly. Researchers push boundaries daily.
    """
    
    strategies = [
        ("Fixed Size", FixedSizeChunker(chunk_size=512)),
        ("Sentence-based", SentenceChunker(target_chunk_size=512)),
        ("Paragraph-based", ParagraphChunker(target_chunk_size=512))
    ]
    
    for strategy_name, chunker in strategies:
        chunks = chunker.chunk(sample_text, "test_doc")
        print(f"\n{strategy_name}: {len(chunks)} chunks")
        for chunk in chunks:
            print(f"  Chunk {chunk['chunk_id']}: {len(chunk['content'])} chars")