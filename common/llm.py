"""Thin wrapper over the local Ollama server that also measures cost.

Every call returns the generated text plus:
  latency_s        wall-clock time of the request
  load_s           time Ollama spent loading the model into RAM (0 when already warm)
  prompt_tokens    tokens the model read (system + context + question)
  output_tokens    tokens the model generated
  tokens_per_s     generation speed reported by Ollama
  peak_rss_mb      peak resident memory of the Ollama runner processes during the call
  avg_cpu_pct      average system CPU utilisation during the call
"""
import threading
import time

import ollama
import psutil

from common.config import GEN_OPTIONS


RUNNER_NAMES = ("ollama", "llama-server")  # Ollama >=0.40 runs model weights in llama-server.exe


def _ollama_procs():
    return [p for p in psutil.process_iter(["name"]) if (p.info["name"] or "").lower().startswith(RUNNER_NAMES)]


class ResourceSampler:
    """Samples Ollama memory and system CPU on a background thread while a request runs."""

    def __init__(self, interval=0.25):
        self.interval = interval
        self.peak_rss = 0
        self.cpu = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        psutil.cpu_percent(None)
        while not self._stop.is_set():
            rss = 0
            for p in _ollama_procs():
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
            self.peak_rss = max(self.peak_rss, rss)
            self.cpu.append(psutil.cpu_percent(None))
            time.sleep(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    @property
    def peak_rss_mb(self):
        return round(self.peak_rss / 2**20, 1)

    @property
    def avg_cpu(self):
        vals = self.cpu[1:] or self.cpu
        return round(sum(vals) / len(vals), 1) if vals else 0.0


def generate(model, prompt, system=None, options=None):
    """Run one chat completion on a local model and return text + measurements."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    opts = {**GEN_OPTIONS, **(options or {})}
    t0 = time.perf_counter()
    with ResourceSampler() as rs:
        r = ollama.chat(model=model, messages=messages, options=opts)
    latency = time.perf_counter() - t0
    eval_ns = r.get("eval_duration") or 0
    out_tok = r.get("eval_count") or 0
    return {
        "text": r["message"]["content"].strip(),
        "latency_s": round(latency, 2),
        "load_s": round((r.get("load_duration") or 0) / 1e9, 2),
        "prompt_tokens": r.get("prompt_eval_count") or 0,
        "output_tokens": out_tok,
        "tokens_per_s": round(out_tok / (eval_ns / 1e9), 2) if eval_ns else 0.0,
        "peak_rss_mb": rs.peak_rss_mb,
        "avg_cpu_pct": rs.avg_cpu,
    }


def unload(model):
    """Free the model's RAM so the next model's memory measurement is not mixed with this one."""
    ollama.generate(model=model, prompt="", keep_alive=0)


def model_info(model):
    """Architecture facts straight from the downloaded model file."""
    s = ollama.show(model)
    d = s.get("details") or {}
    mi = s.get("modelinfo") or s.get("model_info") or {}
    arch = mi.get("general.architecture", d.get("family"))
    return {
        "model": model,
        "family": d.get("family"),
        "parameters": d.get("parameter_size"),
        "quantization": d.get("quantization_level"),
        "format": d.get("format"),
        "layers": mi.get(f"{arch}.block_count"),
        "hidden_size": mi.get(f"{arch}.embedding_length"),
        "attention_heads": mi.get(f"{arch}.attention.head_count"),
        "kv_heads": mi.get(f"{arch}.attention.head_count_kv"),
        "context_length": mi.get(f"{arch}.context_length"),
        "vocab_size": len(mi.get("tokenizer.ggml.tokens") or []) or None,
    }


def model_size_mb(model):
    for m in ollama.list()["models"]:
        if m["model"] == model:
            return round(m["size"] / 2**20, 1)
    return None
