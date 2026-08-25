FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy only the dependency manifest first so Docker's layer cache is keyed on
# requirements.txt, not on every source file - editing app code no longer
# invalidates (and re-downloads) the whole dependency install.
COPY requirements.txt .

# Install CPU-only PyTorch explicitly first. Plain "torch" from PyPI resolves
# to the CUDA/GPU build on Linux, pulling in several GB of NVIDIA libraries
# that are useless here - this container has no GPU access, and the app only
# ever uses PyTorch on CPU (the LLM itself runs via Ollama on the host).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data

EXPOSE 5000

CMD ["python", "backend/backend_server.py"]
