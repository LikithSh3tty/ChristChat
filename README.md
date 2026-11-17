# Student Support AI: LLM, RAG and LLMOps on a laptop CPU

LCA activity (experimental and problem-based learning): implement the three building blocks of a GenAI
application locally, measure them on the same questions, and combine them into a Student Support chatbot
for CHRIST (Deemed to be University). A small PEFT (LoRA) fine-tuning demo is included for comparison.

Everything runs offline after setup. No paid or cloud APIs.

| Model | What it is | Code |
|---|---|---|
| 1. LLM | Local quantised language models (Qwen2.5 1.5B, Llama 3.2 3B) via Ollama, used with no documents | `models/llm/` |
| 2. RAG | Handbook PDF chunked, embedded (MiniLM), stored in ChromaDB, retrieved and passed to the LLM with page citations | `models/rag/` |
| 3. LLMOps | MLflow experiment tracking, versioned prompts, hashed knowledge base, per-query logs and monitoring | `llmops/`, `prompts/`, `experiments/` |
| PEFT demo | LoRA adapter on Qwen2.5 0.5B, base vs fine-tuned under the same RAG setup | `models/peft_lora/` |

Results, interpretation and recommendations are in [report.md](report.md).

## Architecture

```
                     +-------------------- offline indexing (models/rag/ingest.py) --------------------+
 handbook PDF / txt  |  PyMuPDF text per page -> clean headers -> 80-word chunks (20 overlap)             |
 data/documents/ --->|  -> all-MiniLM-L6-v2 embeddings (384-d) -> ChromaDB (cosine) + manifest of hashes   |
                     +--------------------------------------------------------------------------------------+
                                                     |
 student question                                    v
 Streamlit UI  --->  chatbot/service.py  --->  retrieve top-4 chunks (with page numbers)
 chatbot/app.py           |                          |
                          |                          v
                          |                 prompt template (prompts/rag_vN.yaml)
                          |                          |
                          |                          v
                          |                 Ollama local LLM (CPU)  --->  answer + (p. N) citations
                          |                                                     |
                          +--> llmops/monitoring.py: logs/queries.jsonl  <------+
                                (latency, pages, refused, errors)

 experiments/*.py  --->  gold questions (experiments/questions.yaml)  --->  scores  --->  MLflow (mlflow.db)
```

## Requirements

* Windows, macOS or Linux, 8 GB RAM minimum (16 GB used here), about 6 GB free disk
* Python 3.11
* [Ollama](https://ollama.com) 0.40 or newer
* No GPU needed. Measured on an Intel i5-1235U with Iris Xe graphics (CPU inference only)

## Setup

```bash
# 1. Python environment
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Local LLMs (Windows: winget install Ollama.Ollama)
ollama pull qwen2.5:1.5b
ollama pull llama3.2:3b

# 3. Put the CHRIST University handbook PDF (not included in this repo, it is copyrighted)
#    into data/documents/, then build the vector index
python -m models.rag.ingest
```

## Run

```bash
# Chatbot
streamlit run chatbot/app.py

# Model 1: bare LLM, general tasks + closed-book handbook questions
python -m models.llm.run_llm

# Model 2: retrieval sweep (chunk size x top-k), then full RAG
python -m models.rag.run_rag --stage A
python -m models.rag.run_rag --stage B --chunk-words 80 --overlap 20 --k 4 --prompt rag_v1
python -m models.rag.run_rag --stage B --chunk-words 80 --overlap 20 --k 4 --prompt rag_v2

# PEFT demo (downloads Qwen/Qwen2.5-0.5B-Instruct, about 1 GB)
python -m models.peft_lora.make_dataset --n 40
python -m models.peft_lora.train_lora
python -m models.peft_lora.eval_lora

# Model 3: log every result to MLflow, draw charts, view dashboards
python -m experiments.evaluation
python -m experiments.comparison
mlflow ui --backend-store-uri sqlite:///mlflow.db      # http://127.0.0.1:5000
python -m llmops.monitoring                             # chatbot usage summary

# Tests
pytest -m "not integration"     # fast, no LLM
pytest                          # includes real retrieval + LLM calls
```

## Project layout

```
common/            config, Ollama wrapper with latency/memory/CPU measurement, scoring rules
models/llm/        Model 1 experiment
models/rag/        ingest (load, clean, chunk, embed, store), pipeline (retrieve, prompt, generate), experiment
models/peft_lora/  dataset builder, LoRA training, base vs LoRA evaluation, trained adapter
llmops/            MLflow tracking, query logging and monitoring
prompts/           versioned prompt templates (rag_v1, rag_v2)
chatbot/           Streamlit app and service layer
experiments/       gold questions, evaluation, comparison charts, rescoring, results/
tests/             unit and integration tests
data/documents/    knowledge base (university handbook)
```

## How answers are scored

`experiments/questions.yaml` holds 28 questions written by hand from the handbook, each with its gold
answer and PDF page: 15 factual, 3 multi-section, 3 ambiguous, 4 out-of-scope and 3 hallucination traps
(false premises). An answer is correct when it contains every required fact (keyword groups, exact
number matching) and no phrase that contradicts the gold answer. For questions the handbook cannot
answer, only a refusal is correct and any confident answer counts as a hallucination. Scoring is
rule-based so every run is comparable; its limits are discussed in the report.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama is not running` in the app | Start Ollama from the Start menu, or run `ollama serve` |
| `Model ... is not downloaded` | `ollama pull qwen2.5:1.5b` |
| App says the knowledge base is not built | Click **Rebuild index** in the sidebar, or run `python -m models.rag.ingest` |
| `Collection docs_w80_o20 does not exist` | Same as above; the index is not committed to git |
| Very slow first answer | The model loads into RAM on first use (about 10 to 15 s); later answers are faster |
| Out of memory with llama3.2:3b | Use qwen2.5:1.5b, close other apps; 3B needs about 3 GB free RAM |
| Package files missing after install on Windows | Long-path limit; keep the project in a short path or enable Windows long paths |
| `UnicodeEncodeError` in the console | `set PYTHONIOENCODING=utf-8` before running scripts |
