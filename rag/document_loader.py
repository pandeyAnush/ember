import os
from pathlib import Path
from typing import List, Dict

try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


class DocumentLoader:
    """Load documents from TXT, PDF, and DOCX files."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def load_txt(self, file_path: str) -> str:
        """Read text from a .txt file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
            return ""

    def load_pdf(self, file_path: str) -> str:
        """Read text from a PDF file."""
        if not HAS_PDF:
            print("PyPDF2 not installed. Install with: pip install PyPDF2")
            return ""

        try:
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""

    def load_docx(self, file_path: str) -> str:
        """Read text from a DOCX file."""
        if not HAS_DOCX:
            print("python-docx not installed. Install with: pip install python-docx")
            return ""

        try:
            doc = Document(file_path)
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
            return ""

    def load_documents(self) -> List[Dict]:
        """Load all supported documents (.txt, .pdf, .docx, .doc) from data directory."""
        documents = []

        loaders = {
            '.txt': self.load_txt,
            '.pdf': self.load_pdf,
            '.docx': self.load_docx,
            '.doc': self.load_docx,
        }

        for file_path in self.data_dir.iterdir():
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            if ext not in loaders:
                continue

            text = loaders[ext](str(file_path))
            if text.strip():
                documents.append({
                    "content": text,
                    "source": file_path.stem,
                    "file_path": str(file_path),
                    "file_type": ext
                })

        return documents


if __name__ == "__main__":
    loader = DocumentLoader()

    sample_text = """
    Artificial Intelligence is transforming industries worldwide.
    Machine learning models are becoming more sophisticated each year.
    Natural language processing enables computers to understand human text.
    Deep learning has revolutionized computer vision and speech recognition.
    """

    with open("./data/sample.txt", "w") as f:
        f.write(sample_text)

    docs = loader.load_documents()
    print(f"Loaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc['source']}: {len(doc['content'])} characters ({doc['file_type']})")
