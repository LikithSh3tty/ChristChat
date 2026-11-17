"""Single place for paths and default settings shared by all three models."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "documents"
VECTOR_DB_DIR = ROOT / "data" / "vector_db"
RESULTS_DIR = ROOT / "experiments" / "results"
QUESTIONS_FILE = ROOT / "experiments" / "questions.yaml"
PROMPTS_DIR = ROOT / "prompts"
LOG_DIR = ROOT / "logs"

# Local LLMs served by Ollama (quantised GGUF, CPU inference)
LLM_MODELS = ["qwen2.5:1.5b", "llama3.2:3b"]
DEFAULT_LLM = "qwen2.5:1.5b"

# Embedding model for RAG (sentence-transformers, runs on CPU)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Generation settings kept fixed so runs are reproducible
GEN_OPTIONS = {"temperature": 0.0, "seed": 42, "num_predict": 256, "num_ctx": 4096}

for d in (RESULTS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)
