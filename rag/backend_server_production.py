"""
Production Flask Backend with:
- BGE embeddings (1024-dim, with query instruction)
- Cross-encoder reranking
- Llama 3.1 8B LLM (hardware-appropriate for 16GB M4)
- Structured JSON output with metrics
"""

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import sys
from pathlib import Path
import os
import shutil
import json
import re
import numpy as np

# This file lives in rag/, so its own directory is where the sibling
# production/standard modules are - and PROJECT_ROOT (one level up) is
# where data/ and the vector store persist, regardless of the caller's cwd.
sys.path.insert(0, str(Path(__file__).parent))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store_production"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Import production modules
from embeddings_production import ProductionEmbeddingGenerator
from reranker import ProductionReranker
from llm_generator_production import ProductionLLMGenerator
from rag_pipeline_production import ProductionRAGPipeline

# Import standard RAGLab modules
from document_loader import DocumentLoader
from chunking import SentenceChunker
from vector_store import VectorStore
from retriever import Retriever
from evaluation import EvaluationLogger

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB upload cap

# Global variables — heavy models are cached here and loaded exactly once
rag_pipeline = None
evaluation_logger = None
embedding_generator = None
reranker = None
llm_generator = None
chunker = None

# Fingerprint of data/ at the time the persisted index was built, stored
# alongside the vector store so startup can tell whether a rebuild is needed
MANIFEST_PATH = VECTOR_STORE_DIR / "manifest.json"

print("🚀 Production RAGLab Backend Server")
print("=" * 60)


def _data_manifest():
    """Fingerprint data/ as sorted [name, size, mtime] entries. If this matches
    the manifest saved with the vector store, the persisted index is still
    current and we can skip re-embedding entirely."""
    if not DATA_DIR.exists():
        return []
    entries = []
    for f in sorted(DATA_DIR.iterdir()):
        if f.is_file() and not f.name.startswith('.'):
            st = f.stat()
            entries.append([f.name, st.st_size, int(st.st_mtime)])
    return entries


def _is_useful_chunk(text):
    """Filter out noise chunks that pollute retrieval on real PDFs: table-of-
    contents / table-of-figures / table-of-tables lines (dot leaders), number
    tables, separators, and tiny fragments. These are topically similar to
    queries but carry no answerable content, so they crowd out real prose."""
    t = (text or "").strip()
    if len(t) < 80:
        return False
    # TOC / list-of-figures / list-of-tables use dot leaders ("Abstract ...... 4")
    if re.search(r'\.{4,}', t):
        return False
    # mostly non-letters → number tables, separators, dotted rows, arrows
    letters = sum(ch.isalpha() for ch in t)
    if letters < 0.55 * len(t):
        return False
    # needs enough real words to be meaningful context
    if len(re.findall(r'[A-Za-z]{3,}', t)) < 12:
        return False
    # bibliography / reference-list chunks: several "(YYYY)" citations, or an
    # "[Accessed ...]" marker, or multiple URLs — noise for content questions
    if len(re.findall(r'\((?:19|20)\d{2}\)', t)) >= 3:
        return False
    if re.search(r'\[Accessed\b', t) or t.lower().count('http') >= 2:
        return False
    return True


def _chunk_documents(documents):
    """Chunk documents and keep only the useful chunks (drops PDF noise)."""
    chunks = []
    for doc in documents:
        for c in chunker.chunk(doc["content"], doc["source"]):
            if _is_useful_chunk(c["content"]):
                chunks.append(c)
    return chunks


def _ensure_models():
    """Load the heavy models (BGE embeddings, cross-encoder reranker, LLM
    handle, logger, chunker) exactly once and cache them as module globals.
    Re-indexing after an upload/delete reuses these instead of reloading
    gigabytes of model weights every time."""
    global embedding_generator, reranker, llm_generator, evaluation_logger, chunker

    if embedding_generator is None:
        print("1️⃣  Loading BGE Embeddings (1024-dim, with query instruction)...")
        embedding_generator = ProductionEmbeddingGenerator()

    if reranker is None:
        print("2️⃣  Loading Reranker (cross-encoder/ms-marco-MiniLM-L-12-v2)...")
        reranker = ProductionReranker(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")

    if llm_generator is None:
        print("3️⃣  Initializing Llama 3.1 8B (hardware-appropriate for 16GB M4)...")
        llm_generator = ProductionLLMGenerator(model="llama3.1:8b", temperature=0.3)

    if evaluation_logger is None:
        evaluation_logger = EvaluationLogger()

    if chunker is None:
        # Larger chunks (512) give the reranker/LLM more context per hit
        chunker = SentenceChunker(target_chunk_size=512)


def _build_or_load_store(force_rebuild=False):
    """Return (VectorStore, chunk_count).

    Fast path: if a persisted index exists and the data/ fingerprint is
    unchanged, load it straight from disk — no embedding work at all.
    Slow path: (re)embed every document, save the index, and write a fresh
    manifest. Triggered on first build, when data/ changed, or force_rebuild."""
    index_file = VECTOR_STORE_DIR / "faiss.index"
    chunks_file = VECTOR_STORE_DIR / "chunks.json"
    manifest = _data_manifest()

    if (not force_rebuild and index_file.exists() and chunks_file.exists()
            and MANIFEST_PATH.exists()):
        try:
            if json.loads(MANIFEST_PATH.read_text()) == manifest:
                store = VectorStore(embedding_dim=embedding_generator.get_embedding_dimension())
                store.load(str(VECTOR_STORE_DIR))
                print(f"⚡ Reused persisted vector store ({len(store)} chunks) — skipped re-embedding")
                return store, len(store)
            print("♻️  data/ changed since last index — rebuilding vector store")
        except Exception as e:
            print(f"⚠️  Could not reuse saved store ({e}) — rebuilding")

    # Slow path: (re)embed everything
    loader = DocumentLoader(data_dir=str(DATA_DIR))
    documents = loader.load_documents()
    print(f"✅ Loaded {len(documents)} documents" if documents
          else "⚠️  No documents found in data/ folder")

    chunks = _chunk_documents(documents)
    print(f"✅ Created {len(chunks)} chunks (after filtering PDF noise)")

    store = VectorStore(embedding_dim=embedding_generator.get_embedding_dimension())
    if chunks:
        chunk_texts = [c["content"] for c in chunks]
        # Documents get no query-instruction prefix — only queries do
        embeddings = embedding_generator.embed_texts(chunk_texts, is_query=False)
        store.add(embeddings, chunks)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    store.save(str(VECTOR_STORE_DIR))
    MANIFEST_PATH.write_text(json.dumps(manifest))
    print(f"✅ Vector store rebuilt and saved ({len(chunks)} chunks indexed)")
    return store, len(chunks)


def initialize_pipeline(force_rebuild=False):
    """Initialize (or refresh) the production RAG pipeline. Heavy models are
    loaded once and cached; the vector store is loaded from disk when it is
    still current and only re-embedded when data/ changed or force_rebuild."""
    global rag_pipeline

    try:
        print("\n📦 Initializing Production RAG Pipeline...")

        _ensure_models()
        vector_store, chunk_count = _build_or_load_store(force_rebuild=force_rebuild)

        # embedding_model=None: skip loading the unused built-in MiniLM model —
        # production embeds with BGE (embedding_generator) and we assign the
        # BGE-backed vector_store right below.
        retriever = Retriever(chunker, embedding_model=None)
        retriever.vector_store = vector_store

        rag_pipeline = ProductionRAGPipeline(
            retriever=retriever,
            embedding_generator=embedding_generator,
            reranker=reranker,
            llm_generator=llm_generator,
            evaluation_logger=evaluation_logger,
        )

        print("\n" + "=" * 60)
        print("✅ Production RAGLab Backend Ready!")
        print("   Embeddings: BGE (1024-dim) | Reranker: ms-marco-MiniLM-L-12-v2 | LLM: Llama 3.1 8B")
        print(f"   Chunks indexed: {chunk_count}")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Routes
@app.route('/', methods=['GET'])
def index():
    """Serve the RAGLab UI so the whole app is available at one URL
    (http://127.0.0.1:5050) — same-origin, and HTTP reloads always fetch fresh
    (no file:// caching that can leave the browser on stale frontend code)."""
    return send_from_directory(str(FRONTEND_DIR), 'raglab_ui.html')


@app.route('/vendor/<path:filename>', methods=['GET'])
def vendor(filename):
    """Serve the locally vendored JS libraries (React, Babel, Tailwind) so the
    UI has zero external/CDN dependencies."""
    return send_from_directory(str(FRONTEND_DIR / 'vendor'), filename)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "rag_pipeline": rag_pipeline is not None,
        "version": "production-v1.0",
        "model": "llama3.1:8b",
        "embeddings": "BGE (1024-dim)"
    })


@app.route('/query', methods=['POST'])
def query():
    """
    Main query endpoint
    Returns structured JSON with response + metrics
    """
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        if not rag_pipeline:
            return jsonify({"error": "RAG Pipeline not initialized"}), 500
        
        # Execute pipeline
        result = rag_pipeline.query(
            question=question,
            num_initial_retrieval=10,  # Retrieve more, then rerank
            num_final_chunks=3         # Final context size
        )
        
        return jsonify(result)

    except Exception as e:
        print(f"❌ Query error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/query-stream', methods=['POST'])
def query_stream():
    """Streaming query: newline-delimited JSON events (context → tokens → done)
    so the UI shows sources immediately and streams the answer token-by-token."""
    data = request.json or {}
    question = (data.get('question') or '').strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400
    if not rag_pipeline:
        return jsonify({"error": "RAG Pipeline not initialized"}), 500

    def generate():
        try:
            for event in rag_pipeline.query_stream(question,
                                                   num_initial_retrieval=10,
                                                   num_final_chunks=3):
                yield json.dumps(event) + "\n"
        except Exception as e:
            print(f"❌ Stream error: {e}")
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"

    # X-Accel-Buffering off + text/plain keeps tokens flowing without buffering
    return Response(generate(), mimetype='application/x-ndjson',
                    headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})


@app.route('/documents', methods=['GET'])
def get_documents():
    """Get loaded documents"""
    try:
        if not rag_pipeline or not rag_pipeline.retriever:
            return jsonify({"documents": []})

        loader = DocumentLoader(data_dir=str(DATA_DIR))
        documents = loader.load_documents()
        
        return jsonify({
            "count": len(documents),
            "documents": [
                {
                    "source": d["source"], 
                    "size": len(d["content"]),
                    "preview": d["content"][:200]
                } 
                for d in documents
            ]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get pipeline statistics"""
    try:
        if not rag_pipeline:
            return jsonify({"status": "Pipeline not initialized"})
        
        return jsonify({
            "pipeline": "Production with Reranking",
            "embedding_model": "BAAI/bge-large-en-v1.5 (1024-dim)",
            "embedding_features": [
                "Query instruction prefix for optimal retrieval",
                "Batch encoding support",
                "Hardware-optimized"
            ],
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "reranker_features": [
                "Cross-encoder ranking",
                "Direct query-document scoring",
                "Typically 10-15% accuracy improvement"
            ],
            "llm_model": "llama3.1:8b",
            "llm_features": [
                "Hardware-appropriate for 16GB RAM",
                "Better reasoning than Mistral 7B",
                "Reduced hallucinations"
            ],
            "retrieval_pipeline": {
                "initial_retrieval_k": 10,
                "reranking_k": 3,
                "chunk_size": 512
            },
            "quality_metrics": [
                "Retrieval quality (rerank score)",
                "Response relevance",
                "Hallucination detection (heuristic)",
                "Timing breakdowns"
            ]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _add_files_to_store(filenames):
    """Embed ONLY the newly uploaded files (BGE) and append their chunks to the
    live vector store — no re-embedding of existing documents. That full
    re-embed is what OOM-crashed the backend on 16GB during uploads. Returns the
    number of chunks added, or -1 if the store isn't ready (caller should then
    do a one-time full build)."""
    global rag_pipeline
    if (not rag_pipeline or not getattr(rag_pipeline, 'retriever', None)
            or rag_pipeline.retriever.vector_store is None):
        return -1

    stems = {Path(f).stem for f in filenames}
    store = rag_pipeline.retriever.vector_store

    # If any upload replaces an existing file, drop its old chunks first
    # (reconstruct the keepers from the index) so we don't leave duplicates.
    keep_idx = [i for i, c in enumerate(store.chunks) if c.get('source') not in stems]
    if len(keep_idx) != len(store.chunks):
        rebuilt = VectorStore(embedding_dim=store.embedding_dim)
        if keep_idx:
            kv = np.vstack([store.index.reconstruct(int(i)) for i in keep_idx]).astype('float32')
            rebuilt.add(kv, [store.chunks[i] for i in keep_idx])
        store = rebuilt
        rag_pipeline.retriever.vector_store = store

    # Load + chunk ONLY the uploaded files, then embed just those chunks
    loader = DocumentLoader(data_dir=str(DATA_DIR))
    new_docs = [d for d in loader.load_documents() if d.get('source') in stems]
    chunks = _chunk_documents(new_docs)
    if not chunks:
        return 0

    embeddings = embedding_generator.embed_texts([c['content'] for c in chunks], is_query=False)
    store.add(embeddings, chunks)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    store.save(str(VECTOR_STORE_DIR))
    MANIFEST_PATH.write_text(json.dumps(_data_manifest()))
    print(f"⚡ Added {len(chunks)} chunks from {len(new_docs)} new file(s) (no full re-embed)")
    return len(chunks)


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and incrementally index the new files."""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No valid files uploaded'}), 400
        
        # Create data directory if it doesn't exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data_dir_resolved = str(DATA_DIR.resolve())

        uploaded_count = 0
        uploaded_names = []
        for file in files:
            if not file.filename:
                continue

            # secure_filename() strips path separators and '..' components,
            # so a filename like '../../etc/passwd' can't escape data/
            filename = secure_filename(file.filename)
            if not filename:
                continue

            filepath = os.path.join(data_dir_resolved, filename)

            # Defense in depth: confirm the resolved path still lands inside DATA_DIR
            if os.path.commonpath([data_dir_resolved, os.path.abspath(filepath)]) != data_dir_resolved:
                print(f"⚠️  Rejected upload with unsafe filename: {file.filename}")
                continue

            file.save(filepath)
            uploaded_count += 1
            uploaded_names.append(filename)
            print(f"✅ Uploaded: {filename}")

        if uploaded_count == 0:
            return jsonify({'error': 'No valid files could be saved'}), 400

        # Incremental index: embed ONLY the new files and append their chunks —
        # no re-embedding of the whole corpus (that full re-embed is what
        # OOM-crashed the backend on 16GB before).
        print("\n🔄 Indexing new file(s)...")
        added = _add_files_to_store(uploaded_names)
        if added < 0:
            # Store not ready (fresh boot, never indexed) — do the full build once
            if not initialize_pipeline(force_rebuild=True):
                return jsonify({'error': 'Indexing failed'}), 500
        return jsonify({
            'status': 'success',
            'count': uploaded_count,
            'message': f'Uploaded {uploaded_count} file(s) and indexed successfully'
        })
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/list-documents', methods=['GET'])
def list_documents():
    """List all files currently in data/."""
    try:
        files = []
        if DATA_DIR.exists():
            for f in DATA_DIR.iterdir():
                if f.is_file():
                    size = f.stat().st_size
                    files.append({
                        'name': f.name,
                        'size': size,
                        'size_mb': round(size / (1024 * 1024), 2)
                    })
        return jsonify({'count': len(files), 'documents': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _drop_source_from_store(source_stem):
    """Remove one document's chunks from the LIVE vector store without any
    re-embedding: reconstruct the vectors we want to keep straight out of the
    FAISS index, rebuild a fresh index from those, and swap it in. This is what
    makes delete near-instant (no model work) instead of re-embedding every
    remaining document. DocumentLoader stores `source` as the filename stem.
    Returns the number of chunks removed, or -1 if the store isn't ready."""
    global rag_pipeline
    if not rag_pipeline or not getattr(rag_pipeline, 'retriever', None):
        return -1
    store = rag_pipeline.retriever.vector_store
    keep_idx = [i for i, c in enumerate(store.chunks) if c.get('source') != source_stem]
    removed = len(store.chunks) - len(keep_idx)
    if removed <= 0:
        return 0  # this file had no chunks in the index (nothing to do)

    new_store = VectorStore(embedding_dim=store.embedding_dim)
    if keep_idx:
        # IndexFlatL2 supports reconstruct(i) — pull each kept vector back out
        vectors = np.vstack([store.index.reconstruct(int(i)) for i in keep_idx]).astype('float32')
        new_store.add(vectors, [store.chunks[i] for i in keep_idx])

    rag_pipeline.retriever.vector_store = new_store
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    new_store.save(str(VECTOR_STORE_DIR))
    MANIFEST_PATH.write_text(json.dumps(_data_manifest()))
    print(f"⚡ Dropped {removed} chunks for '{source_stem}' (no re-embedding)")
    return removed


@app.route('/delete-document', methods=['POST'])
def delete_document():
    """Delete a document and drop its chunks from the index (no re-embedding)."""
    try:
        filename = (request.json or {}).get('filename')
        if not filename:
            return jsonify({'error': 'Filename required'}), 400

        # secure_filename + confine to DATA_DIR so a crafted name can't escape
        safe = secure_filename(filename)
        filepath = (DATA_DIR / safe).resolve()
        if DATA_DIR.resolve() not in filepath.parents:
            return jsonify({'error': 'Invalid filename'}), 400
        if not filepath.exists():
            return jsonify({'error': f'File not found: {safe}'}), 404

        filepath.unlink()
        print(f"✅ Deleted: {safe}")

        # Fast path: drop this file's chunks from the existing index.
        removed = _drop_source_from_store(Path(safe).stem)

        # Fallback: only if the live store wasn't available (e.g. server just
        # booted and never indexed) do the heavy full rebuild.
        if removed < 0:
            if not initialize_pipeline(force_rebuild=True):
                return jsonify({'error': 'Re-indexing failed'}), 500
            removed = 'all (rebuilt)'

        return jsonify({
            'status': 'success',
            'message': f'Deleted {safe} ({removed} chunks removed)',
            'remaining_documents': len([f for f in DATA_DIR.iterdir() if f.is_file()])
        })

    except Exception as e:
        print(f"❌ Delete error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/clear-all-documents', methods=['POST'])
def clear_all_documents():
    """Delete ALL documents and clear the vector store."""
    try:
        # Remove all files from data/
        if DATA_DIR.exists():
            for f in DATA_DIR.iterdir():
                if f.is_file():
                    f.unlink()
                    print(f"✅ Deleted: {f.name}")

        # The real vector store is a DIRECTORY (faiss.index + chunks.json),
        # not a .pkl file - remove the whole tree
        if VECTOR_STORE_DIR.exists():
            shutil.rmtree(VECTOR_STORE_DIR)
            print("✅ Deleted vector store")

        # Re-initialize empty pipeline (models stay cached)
        if initialize_pipeline(force_rebuild=True):
            return jsonify({
                'status': 'success',
                'message': 'Cleared all documents and vector store',
                'ready_for_new_upload': True
            })
        return jsonify({'error': 'Re-initialization failed'}), 500

    except Exception as e:
        print(f"❌ Clear error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Initialize on startup
    if initialize_pipeline():
        # Port 5050, NOT 5000: macOS Control Center's AirPlay Receiver squats
        # on *:5000 and answers requests with HTTP 403, so a backend on 5000 is
        # shadowed. 5050 is free. (Alternatively disable AirPlay Receiver in
        # System Settings > General > AirDrop & Handoff.)
        print("\n🌐 Starting Flask server on http://127.0.0.1:5050")
        print("⏸️  Press Ctrl+C to stop")
        # threaded=True so a slow request (LLM generation, or a re-index during
        # upload/delete) doesn't block health checks and other requests on the
        # single-threaded dev server — that blocking makes normal slowness look
        # like a crash ("Load failed") in the browser.
        app.run(host='127.0.0.1', port=5050, debug=False, threaded=True)
    else:
        print("\n❌ Failed to initialize. Check errors above.")
        sys.exit(1)