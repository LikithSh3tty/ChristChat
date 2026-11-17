"""PEFT demo, step 3: same 28 gold questions, same RAG retrieval and prompt, base model vs base + LoRA.

The adapter is switched off with `disable_adapter()` for the baseline, so both runs share the exact
same weights except the LoRA update.

Run:  python -m models.peft_lora.eval_lora
"""
import re
import time

import pandas as pd
import psutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.config import RESULTS_DIR
from common.scoring import load_questions, score
from models.peft_lora.train_lora import ADAPTER, BASE
from models.rag.pipeline import RAGConfig, RAGPipeline
from models.rag.run_rag import numbers_grounded, retrieval_metrics


def gen(model, tok, system, user, max_new=160):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    dt = time.perf_counter() - t0
    new = out[0, ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip(), dt, ids.shape[1], len(new)


def main():
    torch.set_num_threads(psutil.cpu_count(logical=False) or 4)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32), ADAPTER).eval()
    pipe = RAGPipeline(RAGConfig(top_k=4, prompt_version="rag_v1"))
    rows = []
    for q in load_questions():
        hits, rt = pipe.retrieve(q["question"])
        user = pipe.build_prompt(q["question"], hits)
        for variant in ("base", "lora"):
            if variant == "base":
                with model.disable_adapter():
                    text, dt, pt, ot = gen(model, tok, pipe.prompt["system"], user)
            else:
                text, dt, pt, ot = gen(model, tok, pipe.prompt["system"], user)
            s = score(q, text)
            rm = retrieval_metrics(q["pages"], [h["page"] for h in hits]) if q.get("pages") else {"hit": None, "rr": None}
            rows.append({"model": f"qwen2.5-0.5b-{variant}", "variant": variant, "id": q["id"], "category": q["category"],
                         "question": q["question"], "gold": q["answer"], "text": text,
                         "cites_page": bool(re.search(r"\(p\.?\s*\d+", text)), "numbers_grounded": numbers_grounded(text, hits),
                         "latency_s": round(dt + rt, 2), "prompt_tokens": pt, "output_tokens": ot,
                         "tokens_per_s": round(ot / dt, 2) if dt else 0.0,
                         "peak_rss_mb": round(psutil.Process().memory_info().rss / 2**20, 1),
                         "avg_cpu_pct": 0.0, **rm, **s})
            print(f"[{q['id']} {variant}] correct={s['correct']} {dt:.1f}s :: {text[:90]!r}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "model4_peft.csv", index=False)
    print(df.groupby("variant").agg(accuracy=("correct", "mean"), hallucination=("hallucinated", "mean"),
                                    refusal=("refused", "mean"), cites_page=("cites_page", "mean"),
                                    latency_s=("latency_s", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
