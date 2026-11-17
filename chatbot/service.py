"""Chatbot service layer: one function the UI and the tests both call.

Keeps the Streamlit file free of RAG/LLMOps details and makes the bot testable without a browser.
"""
import ollama

from common.scoring import is_refusal
from llmops.monitoring import monitored
from models.rag.pipeline import RAGConfig, RAGPipeline


class BackendUnavailable(RuntimeError):
    pass


def check_backend(model: str):
    """Fail early with a readable message when Ollama is down or the model is missing."""
    try:
        names = {m["model"] for m in ollama.list()["models"]}
    except Exception as e:
        raise BackendUnavailable("Ollama is not running. Start it from the Start menu or run `ollama serve`.") from e
    if model not in names:
        raise BackendUnavailable(f"Model `{model}` is not downloaded. Run `ollama pull {model}`.")


def ask(pipe: RAGPipeline, question: str) -> dict:
    """Answer one question with retrieval, generation, citations and monitoring."""
    question = question.strip()
    if not question:
        raise ValueError("Empty question")
    with monitored(question, pipe.cfg.llm, pipe.cfg.prompt_version) as rec:
        r = pipe.answer(question)
        r["not_found"] = is_refusal(r["text"])
        # Only show sources when the answer actually used the documents.
        r["sources"] = [] if r["not_found"] else r["hits"]
        rec.update(answer=r["text"], pages=r["retrieved_pages"], latency_s=r["latency_s"],
                   retrieval_s=r["retrieval_s"], prompt_tokens=r.get("prompt_tokens"),
                   output_tokens=r.get("output_tokens"), top_distance=r["hits"][0]["distance"] if r["hits"] else None,
                   guardrail_refusal=r.get("guardrail_refusal", False))
    return r


def make_pipeline(**kw) -> RAGPipeline:
    return RAGPipeline(RAGConfig(**kw))
