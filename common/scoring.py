"""Rule-based scoring against the hand-written gold set (experiments/questions.yaml).

Scores are deterministic so every run of the experiment produces comparable numbers.
"""
import re

import yaml

from common.config import QUESTIONS_FILE

REFUSAL_PATTERNS = [
    r"could not find", r"couldn't find", r"cannot find", r"can't find", r"not (available|mentioned|covered|specified|provided|found|included|stated)",
    r"no (information|details|mention|data)", r"(does not|doesn't) (contain|mention|include|provide|specify|say)",
    r"i (do not|don't) (know|have)", r"unable to (find|provide|answer)", r"not in the (handbook|documents|context)",
    r"\bnot listed\b", r"\bno [\w ]{0,30} (is |are )?listed\b",
]


def load_questions():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _term_hit(term, text):
    term = term.lower()
    if re.fullmatch(r"\d+", term):
        return re.search(rf"(?<!\d){term}(?!\d)", text) is not None
    return term in text


def keyword_score(text, groups):
    """Fraction of keyword groups found in the answer (1.0 = all required facts present)."""
    if not groups:
        return None
    t = text.lower().replace("’", "'")
    return round(sum(any(_term_hit(term, t) for term in g) for g in groups) / len(groups), 3)


def is_refusal(text):
    t = text.lower().replace("’", "'")
    return any(re.search(p, t) for p in REFUSAL_PATTERNS)


def score(q, text):
    """Return correctness for one answer.

    Answerable question: correct when every keyword group matches.
    Unanswerable question: correct when the model refuses; any confident answer counts as a hallucination.
    """
    refused = is_refusal(text)
    if q.get("expect_refusal"):
        return {"keyword_score": None, "refused": refused, "correct": refused, "hallucinated": not refused}
    ks = keyword_score(text, q.get("keywords"))
    # forbidden = phrases that assert the opposite of the gold answer; any hit makes the answer wrong
    contradicts = any(_term_hit(t, text.lower()) for t in q.get("forbidden", []))
    correct = ks == 1.0 and not contradicts
    return {"keyword_score": ks, "refused": refused, "correct": correct, "hallucinated": (not correct) and not refused}
