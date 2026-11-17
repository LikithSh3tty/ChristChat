"""Log every experiment result to MLflow (Model 3: LLMOps) and print the comparison table.

Each (system, model) pair becomes one MLflow run with params, version hashes, metrics and the per-question CSV.

Run:  python -m experiments.evaluation
View: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import pandas as pd

from common.config import RESULTS_DIR
from llmops.tracking import log_run, summarise
from models.rag.ingest import collection_name

RUNS = [
    # (csv file, system label, prompt version, chunk words, overlap, top_k)
    ("model1_closed_book.csv", "llm_closed_book", None, None, None, None),
    ("model2_rag_rag_v1.csv", "rag", "rag_v1", 80, 20, 4),
    ("model2_rag_rag_v2.csv", "rag", "rag_v2", 80, 20, 4),
    ("model4_peft.csv", "rag_peft", "rag_v1", 80, 20, 4),
]


def main():
    rows = []
    for fname, system, pv, cw, ov, k in RUNS:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"skip {fname} (not run yet)")
            continue
        df = pd.read_csv(path)
        for model, g in df.groupby("model"):
            sub = RESULTS_DIR / f"_{system}_{pv or 'none'}_{model.replace(':', '-')}.csv"
            g.to_csv(sub, index=False)
            params = {"system": system, "llm": model, "prompt_version": pv or "none",
                      "chunk_words": cw, "overlap": ov, "top_k": k}
            m = log_run(f"{system}-{pv or 'none'}-{model}", g, params, sub, prompt_version=pv,
                        index_collection=collection_name(cw, ov) if cw else None)
            sub.unlink()
            rows.append({"system": system, "prompt": pv or "-", "model": model, **m})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "comparison_summary.csv", index=False)
    cols = ["system", "prompt", "model", "accuracy", "hallucination_rate", "refusal_rate", "retrieval_hit_rate",
            "numbers_grounded", "latency_p50_s", "latency_p95_s", "tokens_per_s", "peak_rss_mb"]
    print(table[[c for c in cols if c in table]].to_string(index=False))
    return table


if __name__ == "__main__":
    main()
