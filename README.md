# RAGLab

A small Retrieval-Augmented Generation (RAG) system I built to actually understand how RAG works under the hood, instead of just calling a LangChain one-liner. Everything here — chunking, embeddings, vector search, generation, evaluation — is written from scratch (except for FAISS and Sentence-Transformers doing the heavy lifting), so I could see exactly where things go wrong and why.

It runs fully locally using [Ollama](https://ollama.ai) for the LLM, so no API keys and no cost.

## What it actually does

You give it some text documents, it:

1. **Chunks** them (3 different strategies to compare — fixed-size, sentence-based, paragraph-based)
2. **Embeds** each chunk with `all-MiniLM-L6-v2` (Sentence-Transformers)
3. **Indexes** the embeddings in a FAISS vector store
4. On a query, **retrieves** the most similar chunks
5. Stuffs them into a prompt and **generates** an answer with a local LLM via Ollama
6. **Evaluates** the response with a few heuristics (relevance, possible hallucination, response length, retrieval similarity) and logs everything to a `.jsonl` file so I can track quality over time

There's also a Flask backend + a single-page React frontend (loaded via CDN, no build step) so I can query it from a browser instead of the terminal.

## Project layout

```
RAGLab/
├── main.py              # CLI entry point - runs the full pipeline + interactive Q&A
├── frontend/
│   └── index.html        # Single-file React UI (no npm install needed)
├── backend/
│   └── backend_server.py # Flask API used by the frontend
├── rag/                  # the AI part - retrieval + generation pipeline
│   ├── document_loader.py   # Loads .txt files from data/
│   ├── chunking.py           # FixedSizeChunker / SentenceChunker / ParagraphChunker
│   ├── embeddings.py         # Sentence-Transformers wrapper
│   ├── vector_store.py       # FAISS index + metadata
│   ├── retriever.py          # Ties chunking + embeddings + vector store together
│   ├── llm_generator.py      # Talks to Ollama's /api/generate
│   ├── rag_pipeline.py       # Full retrieve -> generate -> evaluate flow
│   └── evaluation.py         # Precision@k, recall@k, MRR, hallucination heuristic, etc.
├── data/                 # Source documents (.txt) get dropped/created here
├── Dockerfile
├── docker-compose-simple.yml   # Runs the backend in Docker, Ollama stays on the host
└── requirements.txt
```

Frontend, backend, and the AI/RAG pipeline are kept in three separate folders on purpose — `main.py` (CLI) and `backend/backend_server.py` (API) are just two different front doors into the same `rag/` pipeline underneath.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed, with at least one model pulled (this project defaults to `mistral`)

```bash
ollama pull mistral
```

## Running it locally

**1. Create a virtualenv and install dependencies**

macOS's system/Homebrew Python is externally-managed and will refuse a plain `pip install`, so use a venv:

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
```

Use `venv/bin/python` (not bare `python`) for every command below — that's the interpreter that actually has the dependencies installed.

**2. Make sure Ollama is running**

```bash
ollama serve
```

If you get `bind: address already in use` on port 11434, that just means Ollama is already running in the background (it usually auto-starts on macOS/menu bar installs) — you don't need to start it again. Check with:

```bash
curl http://localhost:11434/api/tags
```

If that returns a list of models, you're good.

**3a. Run the CLI version**

```bash
venv/bin/python main.py
```

This creates a couple of sample documents in `data/`, indexes them, runs a handful of test queries, prints a quick evaluation summary, and then drops you into an interactive prompt where you can ask your own questions.

**3b. Or run the web version**

Terminal 1 — start the backend, from the project root (not from inside `backend/`, so it can find the `rag/` and `data/` folders):

```bash
venv/bin/python backend/backend_server.py
```
<!-- http://127.0.0.1:8080 -->
It serves an API on `http://127.0.0.1:5000` (it indexes whatever files are in `data/` on first request).

Terminal 2 — open `frontend/index.html` directly in a browser (`open frontend/index.html` on macOS). It's a static file, no dev server required — but check which port it's pointed at (see note below).

> **Note on ports:** `frontend/index.html` has a single hardcoded backend URL (currently `http://127.0.0.1:5050` — see "Running with Docker" below). If you're running the backend directly via venv on port 5000 instead of through Docker, find-and-replace `5050` → `5000` across `frontend/index.html` first, or the UI will fail to connect.
>
> **Why 127.0.0.1 and not `localhost`:** on macOS, `localhost` resolves to both `127.0.0.1` and `::1`, and macOS's AirPlay Receiver squats on port 5000 over `::1` — so `localhost:5000` can silently hit AirPlay instead of this app. Always use `127.0.0.1` explicitly.

## Running with Docker

If you'd rather keep the backend containerized while Ollama stays on your host machine:

```bash
docker compose -f docker-compose-simple.yml up --build
```

This builds the Flask backend into a container and points it at `host.docker.internal:11434` for Ollama, served on **`http://127.0.0.1:5050`** (not 5000 — the host port is deliberately remapped because macOS's AirPlay Receiver holds port 5000 on all interfaces under System Settings → General → AirDrop & Handoff, which blocks Docker's port publish outright).

It also brings up a second, tiny `nginx` container that just serves `frontend/index.html` as a static file over **`http://127.0.0.1:8080`** — so with Docker, both frontend and backend come up from the one `docker compose up` command; there's no separate "open the HTML file" step like the venv path requires. (The frontend container doesn't talk to the backend container directly — the page still calls `127.0.0.1:5050` from your browser, same as always, it's just served over `http://` instead of `file://` now.)

The Flask API itself has no page at `/` — visiting `http://127.0.0.1:5050` directly in a browser will show Flask's "Not Found" page, that's expected. Go to `http://127.0.0.1:8080` for the actual UI, or hit a real route like `http://127.0.0.1:5050/health` directly to sanity-check the API.

## Production variant (better retrieval, at a cost)

`rag/backend_server_production.py` is a heavier alternative to the standard backend: BGE embeddings (1024-dim instead of 384), a cross-encoder reranking step (retrieve 10 → rerank → keep top 3), and `llama3.1:8b` instead of `mistral`. It's a straight upgrade in answer quality, at the cost of a much slower first startup (downloads the BGE + reranker models, a couple GB) and roughly 2x the latency per query (it does a second LLM call to self-check for hallucinations).

```bash
venv/bin/python rag/backend_server_production.py
```

Also serves on port 5000, so stop the standard backend first if it's running. Same `/health`, `/query`, `/documents`, `/upload` routes as the standard backend, so `frontend/index.html` works against it too (just repoint the port as needed).

## Using your own documents

Drop `.txt` files into `data/` and restart `main.py` or `backend/backend_server.py` — `DocumentLoader` picks up everything in that folder automatically.

## Notes on the evaluation metrics

The generation-side metrics (`has_hallucination`, `relevance`) are simple word-overlap heuristics, not a proper LLM-as-judge setup — good enough to eyeball whether a change to chunking/retrieval helped or hurt, not something I'd trust for a real benchmark. Retrieval metrics (precision@k, recall@k, MRR) in `evaluation.py` need a labeled set of "relevant sources" per query to be useful, which isn't wired up in `main.py` yet.

Every query run through `RAGPipeline.query()` gets appended to `rag_results.jsonl` (or whatever `log_file` you pass in), so results are easy to review after the fact.

## What I was mainly trying to learn

- How chunking strategy affects retrieval quality (there are three chunkers here specifically so I could compare them)
- What FAISS is actually doing under a vector store abstraction
- How much a bad chunk hurts the final generated answer vs. how much the LLM can compensate
- The basics of an "LLMOps" loop — logging every query/response/metric so quality is measurable over time instead of just vibes-checking a few prompts

## Known rough edges

- `requirements.txt` lists `langchain` / `langchain-community` but nothing in the code actually imports them — they were part of an earlier draft. Safe to remove if you want a leaner install.
- `evaluation.py`'s hallucination check is a crude word-overlap heuristic and will misfire on paraphrased-but-correct answers.
- No persistence by default — the vector store rebuilds from `data/` every time you start the app (though `Retriever.save()` / `.load()` exist if you want to wire that up).
