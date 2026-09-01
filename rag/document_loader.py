import os
from pathlib import Path
from typing import List, Dict

try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

# PyMuPDF extracts far cleaner text than PyPDF2 (better spacing, layout, and it
# handles complex/large PDFs where PyPDF2 returns almost nothing). Preferred when
# available; PyPDF2 stays as a fallback.
try:
    import pymupdf as _fitz  # PyMuPDF >= 1.24 exposes the `pymupdf` module name
    HAS_PYMUPDF = True
except ImportError:
    try:
        import fitz as _fitz  # older PyMuPDF
        HAS_PYMUPDF = True
    except ImportError:
        HAS_PYMUPDF = False

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
        """Read text from a PDF. Prefers PyMuPDF (cleaner extraction); falls back
        to PyPDF2 if PyMuPDF is unavailable or returns too little text."""
        text = ""

        # Preferred: PyMuPDF
        if HAS_PYMUPDF:
            try:
                doc = _fitz.open(file_path)
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
            except Exception as e:
                print(f"PyMuPDF failed on {file_path} ({e}); trying PyPDF2")
                text = ""

        # Fallback: PyPDF2 (also used if PyMuPDF extracted almost nothing)
        if len(text.strip()) < 50 and HAS_PDF:
            try:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join((page.extract_text() or "") for page in reader.pages)
            except Exception as e:
                print(f"Error reading PDF {file_path}: {e}")

        if not text.strip():
            print(f" No extractable text from {file_path} (likely a scanned/image PDF)")
        return text

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
