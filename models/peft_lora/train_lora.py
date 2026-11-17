"""PEFT demo, step 2: LoRA fine-tuning of Qwen2.5-0.5B-Instruct on CPU.

LoRA freezes all original weights W and learns a low-rank update  W' = W + (alpha/r) * B @ A,
with A (r x d) and B (d x r). With r = 8 on the attention q/v projections only ~0.1% of
parameters are trained, so it fits a laptop CPU and the result is a few-MB adapter file.

Loss is computed on the assistant answer tokens only (prompt tokens are masked with -100).

Run:  python -m models.peft_lora.train_lora
"""
import json
import time

import psutil
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.config import RESULTS_DIR, ROOT

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
DATA = ROOT / "models" / "peft_lora" / "train.jsonl"
ADAPTER = ROOT / "models" / "peft_lora" / "adapter"
CFG = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"],
       "epochs": 2, "lr": 2e-4, "grad_accum": 4, "max_len": 768, "seed": 42}


def encode(tok, ex):
    msgs = [{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["user"]}]
    prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    ans_ids = tok(ex["assistant"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + ans_ids)[-CFG["max_len"]:]
    labels = ([-100] * len(prompt_ids) + ans_ids)[-CFG["max_len"]:]
    return torch.tensor([ids]), torch.tensor([labels])


def main():
    torch.manual_seed(CFG["seed"])
    torch.set_num_threads(psutil.cpu_count(logical=False) or 4)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    total = sum(p.numel() for p in model.parameters())
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=CFG["r"], lora_alpha=CFG["lora_alpha"],
                                             lora_dropout=CFG["lora_dropout"], target_modules=CFG["target_modules"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {trainable:,} of {total:,} params ({100 * trainable / total:.3f}%)")

    data = [encode(tok, json.loads(l)) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CFG["lr"])
    model.train()
    log, t0, step = [], time.perf_counter(), 0
    for epoch in range(CFG["epochs"]):
        g = torch.Generator().manual_seed(CFG["seed"] + epoch)
        order = torch.randperm(len(data), generator=g).tolist()
        running = []
        for i, idx in enumerate(order, 1):
            ids, labels = data[idx]
            loss = model(input_ids=ids, labels=labels).loss
            (loss / CFG["grad_accum"]).backward()
            running.append(loss.item())
            if i % CFG["grad_accum"] == 0 or i == len(order):
                opt.step()
                opt.zero_grad()
                step += 1
                avg = sum(running) / len(running)
                log.append({"epoch": epoch + 1, "step": step, "loss": round(avg, 4),
                            "elapsed_s": round(time.perf_counter() - t0, 1)})
                print(log[-1])
                running = []
    train_s = time.perf_counter() - t0

    model.save_pretrained(ADAPTER)
    size_mb = sum(f.stat().st_size for f in ADAPTER.rglob("*") if f.is_file()) / 2**20
    meta = {"base": BASE, **CFG, "examples": len(data), "trainable_params": trainable, "total_params": total,
            "trainable_pct": round(100 * trainable / total, 3), "train_s": round(train_s, 1),
            "adapter_mb": round(size_mb, 2), "peak_rss_mb": round(psutil.Process().memory_info().rss / 2**20),
            "first_loss": log[0]["loss"], "final_loss": log[-1]["loss"]}
    (RESULTS_DIR / "peft_train_log.json").write_text(json.dumps({"meta": meta, "steps": log}, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
