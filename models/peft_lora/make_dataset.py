"""PEFT demo, step 1: build a small supervised dataset in the chatbot's RAG format.

Teaches behaviour, not new facts: read the passages, answer briefly with a page citation,
and refuse when the passages do not contain the answer.

  positive  : local LLM (qwen2.5:1.5b) writes one Q/A pair from a handbook chunk; kept only if
              every number in the answer appears in that chunk (no invented facts)
  refusal   : the same generated question paired with passages from other pages -> refusal sentence
  leakage   : any generated question too similar to a gold test question is dropped

Run:  python -m models.peft_lora.make_dataset --n 40
"""
import argparse
import json
import random
import re

import chromadb
import ollama

from common.config import ROOT, VECTOR_DB_DIR
from common.scoring import load_questions
from models.rag.ingest import DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP, collection_name, get_embedder
from models.rag.pipeline import load_prompt

OUT = ROOT / "models" / "peft_lora" / "train.jsonl"
POLICY_PAGES = set(range(18, 23)) | set(range(51, 109))  # library, NEP, assessment, scholarships, regulations
GEN_PROMPT = ("From the passage below, write ONE question a student might ask and its short answer (one sentence) "
              "using only facts in the passage. Reply as JSON: {\"question\": \"...\", \"answer\": \"...\"}\n\nPassage:\n")


def numbers_ok(answer, chunk):
    return all(re.search(rf"(?<!\d){re.escape(n)}(?!\d)", chunk) for n in re.findall(r"\d+(?:\.\d+)?", answer))


def main(n, seed=42, max_gold_sim=0.8):
    random.seed(seed)
    prompt = load_prompt("rag_v1")
    col = chromadb.PersistentClient(path=str(VECTOR_DB_DIR)).get_collection(collection_name(DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP))
    allc = col.get(include=["documents", "metadatas"])
    chunks = [(d, m["page"]) for d, m in zip(allc["documents"], allc["metadatas"]) if m["page"] in POLICY_PAGES]
    random.shuffle(chunks)

    emb = get_embedder()
    gold = emb.encode([q["question"] for q in load_questions()], normalize_embeddings=True)

    def ctx(items):
        return "\n\n".join(f"[{i}] (p. {p}) {t}" for i, (t, p) in enumerate(items, 1))

    rows, dropped = [], {"bad_json": 0, "ungrounded": 0, "too_close_to_test": 0}
    for text, page in chunks:
        if len([r for r in rows if r["kind"] == "positive"]) >= n:
            break
        r = ollama.chat(model="qwen2.5:1.5b", messages=[{"role": "user", "content": GEN_PROMPT + text}],
                        format="json", options={"temperature": 0.3, "seed": seed, "num_predict": 120})
        try:
            qa = json.loads(r["message"]["content"])
            q, a = qa["question"].strip(), qa["answer"].strip()
            assert len(q) > 10 and len(a) > 5
        except Exception:
            dropped["bad_json"] += 1
            continue
        if not numbers_ok(a, text):
            dropped["ungrounded"] += 1
            continue
        qv = emb.encode([q], normalize_embeddings=True)
        if float((gold @ qv.T).max()) > max_gold_sim:
            dropped["too_close_to_test"] += 1
            continue
        # realistic context: what the retriever returns for this question, guaranteed to include the source chunk
        hits = col.query(query_embeddings=qv.tolist(), n_results=4)
        items = [(d, m["page"]) for d, m in zip(hits["documents"][0], hits["metadatas"][0])]
        if text not in [t for t, _ in items]:
            items[-1] = (text, page)
        rows.append({"kind": "positive", "system": prompt["system"],
                     "user": prompt["user"].format(context=ctx(items), question=q),
                     "assistant": f"{a.rstrip('.')} (p. {page})."})
        # matching refusal: same question, passages only from other pages
        others = random.sample([c for c in chunks if c[1] != page], 4)
        if len(rows) % 3 == 0:  # about one refusal per three positives
            rows.append({"kind": "refusal", "system": prompt["system"],
                         "user": prompt["user"].format(context=ctx(others), question=q),
                         "assistant": prompt["refusal"]})
        print(f"[{len(rows)}] p{page} Q: {q[:70]} | A: {a[:60]}")

    random.shuffle(rows)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    kinds = {k: sum(r["kind"] == k for r in rows) for k in ("positive", "refusal")}
    print(json.dumps({"written": len(rows), **kinds, "dropped": dropped}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    main(ap.parse_args().n)
