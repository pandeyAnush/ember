"""
Flask backend server for RAGLab frontend
Connects React UI to the RAG system
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from pathlib import Path
from werkzeug.utils import secure_filename

# Add rag/ (where the pipeline modules actually live) to path
sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))

from retriever import Retriever
from llm_generator import OllamaGenerator
from rag_pipeline import RAGPipeline
from chunking import SentenceChunker
from document_loader import DocumentLoader

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './data'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Global RAG pipeline instance
rag_pipeline = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_rag():
    """Initialize the RAG pipeline"""
    global rag_pipeline
    
    try:
        # Load documents
        loader = DocumentLoader()
        documents = loader.load_documents()
        
        if not documents:
            print("No documents found. Creating sample document...")
            # Create sample
            sample_text = """
            Artificial Intelligence is transforming industries worldwide.
            Machine learning models are becoming more sophisticated each year.
            Deep learning has revolutionized computer vision and speech recognition.
            
            Natural language processing enables computers to understand human text.
            Neural networks are the foundation of modern AI systems.
            Transformers have become the dominant architecture in NLP.
            
            Python is the most popular language for AI and machine learning.
            TensorFlow and PyTorch are leading frameworks for deep learning.
            Data preprocessing is often the most time-consuming part of ML projects.
            """
            Path("./data").mkdir(exist_ok=True)
            with open("./data/sample.txt", "w") as f:
                f.write(sample_text)
            documents = loader.load_documents()
        
        # Initialize retriever
        chunker = SentenceChunker(target_chunk_size=256)
        retriever = Retriever(chunker)
        
        print(f"Indexing {len(documents)} documents...")
        retriever.index_documents(documents)
        
        # Initialize generator
        generator = OllamaGenerator(model="mistral")
        
        # Create pipeline
        rag_pipeline = RAGPipeline(retriever, generator)
        
        print("✓ RAG Pipeline initialized successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error initializing RAG: {e}")
        return False


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'rag_pipeline': rag_pipeline is not None
    })


@app.route('/query', methods=['POST'])
def query():
    """Process a query through the RAG pipeline"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        if rag_pipeline is None:
            return jsonify({'error': 'RAG pipeline not initialized'}), 500
        
        print(f"\n📝 Processing query: {question}")
        
        # Run query through pipeline
        result = rag_pipeline.query(
            question,
            num_context_chunks=3,
            llm_temperature=0.5,
            log=True
        )
        
        # Format response
        response = {
            'question': question,
            'response': result['response'],
            'retrieved_chunks': [
                {
                    'content': chunk['content'],
                    'source': chunk['source'],
                    'similarity': float(chunk['similarity']),
                    'chunk_id': chunk['chunk_id']
                }
                for chunk in result['retrieved_chunks']
            ],
            'metrics': {
                'retrieval': {
                    'num_retrieved': result['metrics']['retrieval']['num_retrieved'],
                    'average_similarity': float(result['metrics']['retrieval']['average_similarity']),
                    'retrieval_time': float(result['metrics']['retrieval']['retrieval_time'])
                },
                'generation': {
                    'response_length': result['metrics']['generation']['response_length'],
                    'has_hallucination': bool(result['metrics']['generation']['has_hallucination']),
                    'relevance': float(result['metrics']['generation']['relevance']),
                    'generation_time': float(result['metrics']['generation']['generation_time'])
                },
                'total_time': float(result['metrics']['total_time'])
            }
        }
        
        print(f"✓ Query processed successfully")
        return jsonify(response)
        
    except Exception as e:
        print(f"✗ Error processing query: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads (.txt, .pdf, .docx, .doc) and re-index documents"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        uploaded_count = 0

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_count += 1
                print(f"✓ Saved: {filename}")

        if uploaded_count == 0:
            return jsonify({'error': 'No valid files uploaded (allowed: .txt, .pdf, .docx, .doc)'}), 400

        global rag_pipeline
        if rag_pipeline is None:
            return jsonify({'error': 'RAG pipeline not initialized'}), 500

        print("Re-indexing documents after upload...")
        loader = DocumentLoader()
        documents = loader.load_documents()

        chunker = SentenceChunker(target_chunk_size=256)
        rag_pipeline.retriever = Retriever(chunker)
        rag_pipeline.retriever.index_documents(documents)

        total_chunks = len(rag_pipeline.retriever.vector_store)

        return jsonify({
            'status': 'success',
            'count': uploaded_count,
            'total_chunks': total_chunks,
            'message': f'Uploaded and indexed {uploaded_count} file(s) with {total_chunks} total chunks'
        })

    except Exception as e:
        print(f"✗ Upload error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/documents', methods=['GET'])
def get_documents():
    """Get info about loaded documents"""
    try:
        if rag_pipeline is None:
            return jsonify({'error': 'RAG pipeline not initialized'}), 500

        loader = DocumentLoader()
        documents = loader.load_documents()
        chunk_count = len(rag_pipeline.retriever.vector_store)

        return jsonify({
            'chunk_count': chunk_count,
            'files': [
                {'source': doc['source'], 'file_type': doc.get('file_type', '.txt')}
                for doc in documents
            ],
            'status': 'ready'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get evaluation statistics"""
    try:
        if rag_pipeline is None:
            return jsonify({'error': 'RAG pipeline not initialized'}), 500
        
        summary = rag_pipeline.get_evaluation_summary()
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.before_request
def before_request():
    """Initialize RAG pipeline on first request"""
    global rag_pipeline
    if rag_pipeline is None:
        print("Initializing RAG pipeline on first request...")
        initialize_rag()


if __name__ == '__main__':
    print("🚀 RAGLab Backend Server Starting...")
    print("=" * 60)
    
    # Try to initialize
    if not initialize_rag():
        print("⚠️  Warning: RAG pipeline failed to initialize")
        print("Make sure Ollama is running: ollama serve")
    
    print("=" * 60)
    print("Server running on http://localhost:5000")
    print("Frontend should connect to this address")
    print("=" * 60)
    
    # Binding to 'localhost' only listens on loopback - fine when run directly
    # on the host, but inside Docker it means Docker's port-forwarded traffic
    # (which arrives via the container's external interface, not loopback)
    # never reaches this socket. FLASK_RUN_HOST=0.0.0.0 in docker-compose
    # overrides this for the containerized case.
    app.run(debug=True, host=os.environ.get('FLASK_RUN_HOST', 'localhost'), port=5000)
