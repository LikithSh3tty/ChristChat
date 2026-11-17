"""MODEL 2: Retrieval-Augmented Generation over the university handbook.

Stage A (retrieval only, no LLM): sweep chunk size and top-k, measure whether the gold page is retrieved.
Stage B (full RAG): best config, both LLMs, same 28 gold questions as Model 1, scored the same way.

Run:  python -m models.rag.run_rag
      python -m models.rag.run_rag --stage A
"""
import argparse
import re

import pandas as pd

from common.config import LLM_MODELS, RESULTS_DIR
from common.llm import unload
from common.scoring import load_questions, score
from models.rag.ingest import build_index
from models.rag.pipeline import RAGConfig, RAGPipeline

CHUNK_CONFIGS = [(80, 20), (120, 30), (200, 40)]
TOP_KS = [1, 2, 4, 6]


def retrieval_metrics(gold_pages, retrieved_pages):
    """hit = any gold page in the retrieved set; rr = 1/rank of first gold page (0 if none)."""
    gold = set(gold_pages)
    rank = next((i for i, p in enumerate(retrieved_pages, 1) if p in gold), None)
    return {"hit": rank is not None, "rr": 1 / rank if rank else 0.0}


def numbers_grounded(answer, hits):
    """Faithfulness proxy: share of numbers in the answer that also appear in the retrieved context.
    Page citations like (p. 21) are excluded. 1.0 means no number was invented."""
    text = re.sub(r"\(p+\.?\s*\d+(,\s*\d+)*\)", "", answer)
    nums = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?", text))
    if not nums:
        return None
    ctx = " ".join(h["text"] for h in hits)
    return round(sum(re.search(rf"(?<![\d]){re.escape(n)}(?![\d])", ctx) is not None for n in nums) / len(nums), 3)


def stage_a(questions):
    answerable = [q for q in questions if q.get("pages")]
    rows, index_rows = [], []
    for cw, ov in CHUNK_CONFIGS:
        stats = build_index(cw, ov)
        index_rows.append(stats | {"files": ",".join(stats["files"])})
        pipe = RAGPipeline(RAGConfig(chunk_words=cw, overlap=ov))
        for q in answerable:
            hits, rt = pipe.retrieve(q["question"], k=max(TOP_KS))
            pages = [h["page"] for h in hits]
            for k in TOP_KS:
                rows.append({"chunk_words": cw, "overlap": ov, "k": k, "id": q["id"], "category": q["category"],
                             "retrieval_s": rt, **retrieval_metrics(q["pages"], pages[:k])})
        print(f"indexed w{cw}/o{ov}: {stats['chunks']} chunks, embed {stats['embed_s']}s")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "model2_retrieval_sweep.csv", index=False)
    pd.DataFrame(index_rows).to_csv(RESULTS_DIR / "model2_index_stats.csv", index=False)
    summary = df.groupby(["chunk_words", "k"]).agg(hit_rate=("hit", "mean"), mrr=("rr", "mean"),
                                                   retrieval_s=("retrieval_s", "mean")).round(3)
    print("\n=== Retrieval sweep (measured, answerable questions) ===\n", summary)
    return summary


def stage_b(questions, chunk_words, overlap, k, models, prompt_version="rag_v1"):
    rows = []
    for m in models:
        pipe = RAGPipeline(RAGConfig(llm=m, chunk_words=chunk_words, overlap=overlap, top_k=k, prompt_version=prompt_version))
        pipe.answer("warm up")  # load model weights before timing
        for q in questions:
            r = pipe.answer(q["question"])
            s = score(q, r["text"])
            rm = retrieval_metrics(q["pages"], r["retrieved_pages"]) if q.get("pages") else {"hit": None, "rr": None}
            rows.append({"model": m, "id": q["id"], "category": q["category"], "question": q["question"], "gold": q["answer"],
                         "text": r["text"], "retrieved_pages": r["retrieved_pages"], "top_distance": r["hits"][0]["distance"],
                         "cites_page": bool(re.search(r"\(p\.?\s*\d+", r["text"])),
                         "numbers_grounded": numbers_grounded(r["text"], r["hits"]),
                         **{k_: r[k_] for k_ in ("latency_s", "retrieval_s", "prompt_tokens", "output_tokens",
                                                 "tokens_per_s", "peak_rss_mb", "avg_cpu_pct")}, **rm, **s})
            print(f"[{m} {q['id']}] correct={s['correct']} hit={rm['hit']} {r['latency_s']}s :: {r['text'][:90]!r}")
        unload(m)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"model2_rag_{prompt_version}.csv", index=False)
    print("\n=== RAG summary (measured) ===")
    print(df.groupby("model").agg(accuracy=("correct", "mean"), hallucination_rate=("hallucinated", "mean"),
                                  refusal_rate=("refused", "mean"), hit_rate=("hit", "mean"),
                                  numbers_grounded=("numbers_grounded", "mean"), mean_latency_s=("latency_s", "mean"),
                                  peak_rss_mb=("peak_rss_mb", "max")).round(3).to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B", "AB"], default="AB")
    ap.add_argument("--chunk-words", type=int, default=120)
    ap.add_argument("--overlap", type=int, default=30)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--models", nargs="+", default=LLM_MODELS)
    ap.add_argument("--prompt", default="rag_v1")
    a = ap.parse_args()
    qs = load_questions()
    if "A" in a.stage:
        stage_a(qs)
    if "B" in a.stage:
        stage_b(qs, a.chunk_words, a.overlap, a.k, a.models, a.prompt)
