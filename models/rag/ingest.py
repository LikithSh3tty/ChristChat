"""RAG step 1-5: load documents, clean text, chunk, embed, store in ChromaDB.

Each chunk keeps its source file and PDF page number so answers can cite pages.

Run:  python -m models.rag.ingest                       (default chunk size)
      python -m models.rag.ingest --chunk-words 200 --overlap 40
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import chromadb
import pymupdf as fitz  # PyMuPDF

from common.config import DOCS_DIR, EMBED_MODEL, VECTOR_DB_DIR

# Chosen from the Stage A sweep (experiments/results/model2_retrieval_sweep.csv):
# 80-word chunks gave the best hit@4 (0.957) and MRR (0.808) on the gold set.
DEFAULT_CHUNK_WORDS = 80
DEFAULT_OVERLAP = 20

# Running headers/footers repeated on every handbook page; they add noise to every chunk.
BOILERPLATE = [r"^Handbook 2026-27\s*$", r"^CHRIST \(Deemed to be University\)\s*$", r"^\d{1,3}\s*$"]

_embedder = None


def get_embedder():
    """Loaded once per process; all-MiniLM-L6-v2 maps text to a 384-dim vector."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _embedder


def load_pages(path: Path):
    """Yield (page_number, text) for PDF or plain-text files."""
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                yield i + 1, page.get_text()
    else:
        yield 1, path.read_text(encoding="utf-8", errors="ignore")


def clean(text: str) -> str:
    lines = [l for l in text.splitlines() if not any(re.match(p, l.strip()) for p in BOILERPLATE)]
    text = "\n".join(lines)
    text = re.sub(r"-\n(?=[a-z])", "", text)        # re-join hyphenated line breaks
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)      # PDF line breaks are layout, not meaning
    return re.sub(r"\s{2,}", " ", text).strip()


def chunk_words(text: str, size: int, overlap: int):
    """Sliding window over words. Overlap keeps a sentence that straddles a boundary in both chunks."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step)]


def collection_name(chunk_words_: int, overlap: int) -> str:
    return f"docs_w{chunk_words_}_o{overlap}"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_index(chunk_words_=DEFAULT_CHUNK_WORDS, overlap=DEFAULT_OVERLAP, files=None, reset=True):
    files = files or sorted(p for p in DOCS_DIR.iterdir() if p.suffix.lower() in {".pdf", ".txt", ".md"})
    t0 = time.perf_counter()
    ids, docs, metas = [], [], []
    for f in files:
        for page, raw in load_pages(f):
            for j, c in enumerate(chunk_words(clean(raw), chunk_words_, overlap)):
                if len(c.split()) < 8:  # skip near-empty fragments (cover pages, blank pages)
                    continue
                ids.append(f"{f.stem}:p{page}:c{j}")
                docs.append(c)
                metas.append({"source": f.name, "page": page, "chunk": j})
    t_chunk = time.perf_counter() - t0

    t1 = time.perf_counter()
    vectors = get_embedder().encode(docs, batch_size=32, normalize_embeddings=True, show_progress_bar=False).tolist()
    t_embed = time.perf_counter() - t1

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    name = collection_name(chunk_words_, overlap)
    if reset and name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(ids), 500):
        col.upsert(ids=ids[i:i + 500], documents=docs[i:i + 500], embeddings=vectors[i:i + 500], metadatas=metas[i:i + 500])

    stats = {
        "collection": name, "chunk_words": chunk_words_, "overlap": overlap, "embed_model": EMBED_MODEL,
        "files": {f.name: file_hash(f) for f in files}, "chunks": len(ids),
        "avg_chunk_words": round(sum(len(d.split()) for d in docs) / max(1, len(docs)), 1),
        "chunk_s": round(t_chunk, 2), "embed_s": round(t_embed, 2),
    }
    (VECTOR_DB_DIR / f"{name}_manifest.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    a = ap.parse_args()
    print(json.dumps(build_index(a.chunk_words, a.overlap), indent=2))
