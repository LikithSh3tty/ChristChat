"""MODEL 1: the bare LLM (no retrieval).

Two experiments per local model:
  A. General GenAI tasks (summarise, explain, rewrite, classify, reason) show what a small
     local LLM can do with only its pre-trained knowledge.
  B. Closed-book handbook questions: the same gold questions used later for RAG. The model has
     never seen the Christ University handbook, so this measures how often it invents policy.

Run:  python -m models.llm.run_llm            (both models)
      python -m models.llm.run_llm --models qwen2.5:1.5b --limit 5
"""
import argparse
import json
import time

import pandas as pd

from common.config import LLM_MODELS, RESULTS_DIR
from common.llm import generate, model_info, model_size_mb, unload
from common.scoring import load_questions, score

SYSTEM = "You are a helpful student support assistant for CHRIST (Deemed to be University), Bangalore. Answer concisely."

GENERAL_TASKS = [
    ("summarise", "Summarise in two sentences: Retrieval-Augmented Generation (RAG) combines a retriever that "
                  "fetches relevant passages from a document collection with a generator model that writes an "
                  "answer conditioned on those passages. This lets the system answer questions about private or "
                  "recent documents and reduces hallucination, at the cost of extra latency and dependence on retrieval quality."),
    ("explain", "Explain to a first-year student, in under 80 words, what continuous internal assessment means."),
    ("rewrite", "Rewrite this as a polite email to a professor: 'i was sick, missed the cia test, can i redo it'"),
    ("classify", "Classify this student query into exactly one of [Exams, Library, Fees, Attendance, Other] and reply "
                 "with the label only: 'How long can I keep a borrowed book?'"),
    ("reason", "A student attended 51 of 60 classes. What is their attendance percentage? Show the calculation."),
]


def run(models, limit=None):
    questions = load_questions()[:limit] if limit else load_questions()
    info, general_rows, qa_rows = [], [], []

    for m in models:
        print(f"\n=== {m} ===")
        t0 = time.perf_counter()
        warm = generate(m, "Say OK.", options={"num_predict": 4})  # cold start: loads weights into RAM
        meta = model_info(m) | {"disk_mb": model_size_mb(m), "cold_load_s": warm["load_s"],
                                "cold_first_call_s": round(time.perf_counter() - t0, 2), "warm_rss_mb": warm["peak_rss_mb"]}
        info.append(meta)
        print(json.dumps(meta, indent=1))

        for task, prompt in GENERAL_TASKS:
            r = generate(m, prompt, system=SYSTEM)
            general_rows.append({"model": m, "task": task, "prompt": prompt, **r})
            print(f"[{task}] {r['latency_s']}s {r['tokens_per_s']} tok/s :: {r['text'][:120]!r}")

        for q in questions:
            r = generate(m, q["question"], system=SYSTEM)
            s = score(q, r["text"])
            qa_rows.append({"model": m, "id": q["id"], "category": q["category"], "question": q["question"],
                            "gold": q["answer"], **r, **s})
            print(f"[{q['id']}] correct={s['correct']} halluc={s['hallucinated']} {r['latency_s']}s :: {r['text'][:100]!r}")
        unload(m)

    with open(RESULTS_DIR / "model1_model_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    pd.DataFrame(general_rows).to_csv(RESULTS_DIR / "model1_general_tasks.csv", index=False)
    qa = pd.DataFrame(qa_rows)
    qa.to_csv(RESULTS_DIR / "model1_closed_book.csv", index=False)

    print("\n=== Closed-book summary (measured) ===")
    print(qa.groupby("model").agg(accuracy=("correct", "mean"), hallucination_rate=("hallucinated", "mean"),
                                  refusal_rate=("refused", "mean"), mean_latency_s=("latency_s", "mean"),
                                  tokens_per_s=("tokens_per_s", "mean"), peak_rss_mb=("peak_rss_mb", "max")).round(3))
    return qa


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=LLM_MODELS)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    run(a.models, a.limit)
