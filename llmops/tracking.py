"""MODEL 3 (LLMOps): experiment tracking and versioning with MLflow.

Every evaluation run is logged with everything needed to reproduce it:
  params    LLM, prompt version, chunking, top-k, generation options, embedding model
  versions  git commit, SHA-256 of prompt file, gold question file and indexed documents
  metrics   accuracy, hallucination/refusal rate, retrieval hit rate, latency p50/p95, memory
  artifacts the per-question CSV and the prompt file itself

View:  mlflow ui --backend-store-uri sqlite:///mlflow.db   then open http://127.0.0.1:5000
"""
import hashlib
import json
import subprocess
from pathlib import Path

import mlflow
import pandas as pd

from common.config import EMBED_MODEL, GEN_OPTIONS, PROMPTS_DIR, QUESTIONS_FILE, ROOT, VECTOR_DB_DIR

TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT = "student-support-ai"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "uncommitted"
    except OSError:
        return "unknown"


def setup():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)


def summarise(df: pd.DataFrame) -> dict:
    """Aggregate per-question results into run-level metrics."""
    m = {
        "accuracy": df["correct"].mean(),
        "hallucination_rate": df["hallucinated"].mean(),
        "refusal_rate": df["refused"].mean(),
        "latency_mean_s": df["latency_s"].mean(),
        "latency_p50_s": df["latency_s"].quantile(0.5),
        "latency_p95_s": df["latency_s"].quantile(0.95),
        "tokens_per_s": df["tokens_per_s"].replace(0, pd.NA).dropna().mean(),
        "peak_rss_mb": df["peak_rss_mb"].max(),
        "avg_cpu_pct": df["avg_cpu_pct"].mean(),
    }
    if "hit" in df and df["hit"].notna().any():
        m["retrieval_hit_rate"] = df["hit"].dropna().astype(float).mean()
        m["retrieval_mrr"] = df["rr"].dropna().mean()
    if "numbers_grounded" in df and df["numbers_grounded"].notna().any():
        m["numbers_grounded"] = df["numbers_grounded"].dropna().mean()
    for cat, g in df.groupby("category"):
        m[f"acc_{cat}"] = g["correct"].mean()
    if "error" in df:
        m["error_rate"] = df["error"].notna().mean()
    return {k: round(float(v), 4) for k, v in m.items() if pd.notna(v)}


def log_run(run_name: str, df: pd.DataFrame, params: dict, csv_path: Path, prompt_version: str | None = None,
            index_collection: str | None = None, tags: dict | None = None) -> dict:
    setup()
    metrics = summarise(df)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"git_commit": git_commit(), "questions_sha": sha(QUESTIONS_FILE), **(tags or {})})
        mlflow.log_params({**params, **{f"gen_{k}": v for k, v in GEN_OPTIONS.items()}, "embed_model": EMBED_MODEL,
                           "n_questions": len(df) // max(1, df["model"].nunique())})
        if prompt_version:
            pf = PROMPTS_DIR / f"{prompt_version}.yaml"
            mlflow.log_param("prompt_sha", sha(pf))
            mlflow.log_artifact(str(pf), "prompt")
        if index_collection:
            mf = VECTOR_DB_DIR / f"{index_collection}_manifest.json"
            if mf.exists():
                mlflow.log_param("kb_files", json.dumps(json.loads(mf.read_text())["files"]))
                mlflow.log_artifact(str(mf), "knowledge_base")
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(csv_path), "results")
    return metrics
