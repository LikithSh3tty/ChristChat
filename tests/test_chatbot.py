"""Tests for the Student Support AI chatbot.

Fast unit tests (no LLM):          pytest -m "not integration"
Full suite incl. local LLM calls:  pytest
Integration tests need Ollama running with qwen2.5:1.5b and the index built (python -m models.rag.ingest).
"""
import pytest

from common.scoring import is_refusal, keyword_score, load_questions, score
from models.rag.ingest import chunk_words, clean
from models.rag.run_rag import numbers_grounded, retrieval_metrics

QS = {q["id"]: q for q in load_questions()}


# ---------------- unit: data preparation ----------------
def test_clean_removes_running_headers_and_joins_lines():
    raw = "Handbook 2026-27\n54\nStudents must secure\nminimum 85% attendance.\nCHRIST (Deemed to be University)\n"
    assert clean(raw) == "Students must secure minimum 85% attendance."


def test_chunks_overlap_and_cover_all_words():
    words = [f"w{i}" for i in range(100)]
    chunks = chunk_words(" ".join(words), size=40, overlap=10)
    assert all(len(c.split()) <= 40 for c in chunks)
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]
    assert set(" ".join(chunks).split()) == set(words)


# ---------------- unit: scoring ----------------
def test_numbers_match_on_digit_boundaries():
    assert keyword_score("fine of Rs. 5 per day", [["5"]]) == 1.0
    assert keyword_score("fine of Rs. 50 per day", [["5"]]) == 0.0


def test_refusal_detection():
    assert is_refusal("I could not find this in the university documents.")
    assert not is_refusal("The minimum attendance is 85% (p. 56).")


def test_out_of_scope_answer_counts_as_hallucination():
    s = score(QS["Q22"], "The hostel fee is Rs 90,000 per year.")
    assert s["hallucinated"] and not s["correct"]


def test_false_premise_must_be_corrected():
    assert not score(QS["Q26"], "Students need 75% attendance.")["correct"]
    assert score(QS["Q26"], "The handbook requires 85%, not 75% (p. 56).")["correct"]


def test_retrieval_metrics():
    assert retrieval_metrics([21], [18, 21, 22]) == {"hit": True, "rr": 0.5}
    assert retrieval_metrics([21], [18, 19]) == {"hit": False, "rr": 0.0}


def test_numbers_grounded_ignores_page_citations():
    hits = [{"text": "Minimum 85% aggregate attendance"}]
    assert numbers_grounded("You need 85% (p. 56).", hits) == 1.0
    assert numbers_grounded("You need 75%.", hits) == 0.0


def test_monitoring_summary_without_errors():
    import pandas as pd
    from llmops.monitoring import summary
    df = pd.DataFrame([{"question": "q", "model": "m", "wall_s": 1.0, "refused": False}])
    s = summary(df)
    assert s["requests"] == 1 and s["error_rate"] == 0 and s["recent_errors"] == []


def test_gold_set_shape():
    assert len(QS) >= 20
    cats = {q["category"] for q in QS.values()}
    assert {"factual", "multi_section", "ambiguous", "out_of_scope", "hallucination_trap"} <= cats


# ---------------- integration: real retrieval + local LLM ----------------
@pytest.fixture(scope="module")
def pipe():
    from chatbot.service import BackendUnavailable, check_backend, make_pipeline
    try:
        check_backend("qwen2.5:1.5b")
        return make_pipeline(llm="qwen2.5:1.5b")
    except (BackendUnavailable, Exception) as e:  # missing index or Ollama
        pytest.skip(f"backend not available: {e}")


@pytest.mark.integration
def test_retrieval_finds_attendance_page(pipe):
    hits, _ = pipe.retrieve("minimum attendance for end semester exam")
    assert {56, 58} & {h["page"] for h in hits}


@pytest.mark.integration
def test_answers_factual_question_with_citation(pipe):
    from chatbot.service import ask
    r = ask(pipe, QS["Q01"]["question"])
    assert score(QS["Q01"], r["text"])["correct"], r["text"]
    assert r["sources"], "answer should carry source passages"


@pytest.mark.integration
def test_refuses_question_not_in_handbook(pipe):
    from chatbot.service import ask
    r = ask(pipe, QS["Q23"]["question"])
    assert r["not_found"], r["text"]
    assert r["sources"] == []


@pytest.mark.integration
def test_empty_question_rejected(pipe):
    from chatbot.service import ask
    with pytest.raises(ValueError):
        ask(pipe, "   ")
