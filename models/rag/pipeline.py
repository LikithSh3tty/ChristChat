"""RAG steps 6-9: retrieve relevant chunks, build a grounded prompt, generate, return citations.

    question --embed--> query vector --cosine search in ChromaDB--> top-k chunks (with pages)
             --prompt template (versioned YAML)--> local LLM --> answer + cited pages
"""
import time
from dataclasses import dataclass, field

import chromadb
import yaml

from common.config import DEFAULT_LLM, PROMPTS_DIR, VECTOR_DB_DIR
from common.llm import generate
from models.rag.ingest import DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP, collection_name, get_embedder


def load_prompt(version: str) -> dict:
    with open(PROMPTS_DIR / f"{version}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class RAGConfig:
    llm: str = DEFAULT_LLM
    prompt_version: str = "rag_v1"
    chunk_words: int = DEFAULT_CHUNK_WORDS
    overlap: int = DEFAULT_OVERLAP
    top_k: int = 4
    # Guardrail: if even the best chunk is this far away (cosine distance), skip the LLM and refuse.
    # None disables it.
    max_distance: float | None = None
    extra: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, cfg: RAGConfig = RAGConfig()):
        self.cfg = cfg
        self.prompt = load_prompt(cfg.prompt_version)
        client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        self.col = client.get_collection(collection_name(cfg.chunk_words, cfg.overlap))
        self.embedder = get_embedder()

    def retrieve(self, question: str, k: int | None = None):
        t0 = time.perf_counter()
        qv = self.embedder.encode([question], normalize_embeddings=True).tolist()
        r = self.col.query(query_embeddings=qv, n_results=k or self.cfg.top_k)
        hits = [{"text": d, "page": m["page"], "source": m["source"], "distance": round(dist, 4)}
                for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]
        return hits, round(time.perf_counter() - t0, 3)

    def build_prompt(self, question, hits):
        context = "\n\n".join(f"[{i}] (p. {h['page']}) {h['text']}" for i, h in enumerate(hits, 1))
        return self.prompt["user"].format(context=context, question=question)

    def answer(self, question: str) -> dict:
        hits, retrieval_s = self.retrieve(question)
        base = {"question": question, "hits": hits, "retrieved_pages": [h["page"] for h in hits],
                "retrieval_s": retrieval_s, "llm": self.cfg.llm, "prompt_version": self.cfg.prompt_version}
        if self.cfg.max_distance is not None and (not hits or hits[0]["distance"] > self.cfg.max_distance):
            return base | {"text": self.prompt["refusal"], "guardrail_refusal": True, "latency_s": retrieval_s,
                           "prompt_tokens": 0, "output_tokens": 0, "tokens_per_s": 0.0, "peak_rss_mb": 0.0,
                           "avg_cpu_pct": 0.0, "load_s": 0.0}
        g = generate(self.cfg.llm, self.build_prompt(question, hits), system=self.prompt["system"])
        g["latency_s"] = round(g["latency_s"] + retrieval_s, 2)
        return base | g | {"guardrail_refusal": False}
