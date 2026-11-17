"""MODEL 3 (LLMOps): production-style logging and monitoring for the chatbot.

Every chatbot request is appended as one JSON line to logs/queries.jsonl:
  timestamp, question, model, prompt version, retrieved pages, latency, tokens, refused, error
`summary()` turns that log into the numbers an operator watches: volume, latency p50/p95,
error rate, refusal rate (knowledge-base gaps) and the most frequent unanswered questions.

Run:  python -m llmops.monitoring       (prints the current summary)
"""
import json
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from common.config import LOG_DIR
from common.scoring import is_refusal

QUERY_LOG = LOG_DIR / "queries.jsonl"


def log_query(record: dict):
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    with open(QUERY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@contextmanager
def monitored(question: str, model: str, prompt_version: str):
    """Wrap one chatbot request; logs success or the error, never swallows it."""
    rec = {"question": question, "model": model, "prompt_version": prompt_version}
    t0 = time.perf_counter()
    try:
        yield rec
        rec["refused"] = is_refusal(rec.get("answer", ""))
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc(limit=3)
        raise
    finally:
        rec["wall_s"] = round(time.perf_counter() - t0, 2)
        log_query(rec)


def load_log() -> pd.DataFrame:
    if not QUERY_LOG.exists():
        return pd.DataFrame()
    return pd.read_json(QUERY_LOG, lines=True)


def summary(df: pd.DataFrame | None = None) -> dict:
    df = load_log() if df is None else df
    if df.empty:
        return {"requests": 0}
    err = df["error"].notna() if "error" in df else pd.Series(False, index=df.index)
    ok = df[~err]
    refused = ok["refused"].fillna(False).astype(bool) if "refused" in ok else pd.Series(False, index=ok.index)
    return {
        "requests": int(len(df)),
        "error_rate": round(float(err.mean()), 3),
        "refusal_rate": round(float(refused.mean()), 3) if len(ok) else 0.0,
        "latency_p50_s": round(float(ok["wall_s"].quantile(0.5)), 2) if len(ok) else None,
        "latency_p95_s": round(float(ok["wall_s"].quantile(0.95)), 2) if len(ok) else None,
        "by_model": ok.groupby("model")["wall_s"].mean().round(2).to_dict() if len(ok) else {},
        "top_unanswered": ok[refused]["question"].str.lower().value_counts().head(5).to_dict(),
        "recent_errors": df[err]["error"].tail(3).tolist() if "error" in df else [],
    }


if __name__ == "__main__":
    print(json.dumps(summary(), indent=2, ensure_ascii=False))
