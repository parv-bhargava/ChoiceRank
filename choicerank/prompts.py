"""Typed relevance questions shared by Jev / Clef / Jeff (same Decisions schema).

Each question type maps a typed answer to (rank score, P(relevant)):
- choice3: options {relevant, partially, not_relevant}; P(rel) = p[relevant];
  rank score = expected grade (2*p_rel + 1*p_partial) / 2, ties broken by p_rel.
- noul:    P(yes) is both the rank score and P(rel).
- score4:  4-level scale (0..3, mirrors TREC DL grading); rank score = expected level / 3,
  P(rel) = P(level >= 2).
"""

from __future__ import annotations

QUESTIONS: dict[str, dict] = {
    "choice3": {
        "type": "choice",
        "instructions": "How relevant is the passage to the search query?",
        "criteria": {
            "relevant": "The passage directly answers or fully addresses the query.",
            "partially": "The passage is on topic and partly addresses the query, but is incomplete or indirect.",
            "not_relevant": "The passage does not address the query.",
        },
    },
    "noul": {
        "type": "noul",
        "instructions": "Does the passage answer or address the search query?",
        "criteria": {
            "true": "The passage answers or substantively addresses the query.",
            "false": "The passage is off topic or does not help answer the query.",
        },
    },
    "score4": {
        "type": "score",
        "instructions": "How relevant is the passage to the search query?",
        "criteria": [
            "Irrelevant: the passage has nothing to do with the query.",
            "Related: on topic but does not answer the query.",
            "Highly relevant: partly answers the query.",
            "Perfectly relevant: dedicated to the query and fully answers it.",
        ],
    },
}


# Prompt-sensitivity paraphrases of the Noul question (issue #12). Fixed before any paraphrase run; same readout as noul.
PARAPHRASES: dict[str, dict] = {
    "noul_p1": {
        "type": "noul",
        "instructions": "Is the passage relevant to the search query?",
        "criteria": {"true": "The passage is relevant to the query.",
                     "false": "The passage is not relevant to the query."},
    },
    "noul_p2": {
        "type": "noul",
        "instructions": "Would this passage be useful to someone who issued the search query?",
        "criteria": {"true": "The passage gives information that helps with the query.",
                     "false": "The passage gives no information that helps with the query."},
    },
    "noul_p3": {
        "type": "noul",
        "instructions": "Does the passage contain the information the search query is looking for?",
        "criteria": {"true": "The passage contains the information sought by the query.",
                     "false": "The passage does not contain the information sought by the query."},
    },
}


def make_state(query: str, passage: str) -> dict:
    return {"query": query, "passage": passage}


def questions_for(qtype: str) -> dict:
    """`qtype` is one question id or 'all' (all three in one request; input-only cost)."""
    if qtype == "all":
        return dict(QUESTIONS)
    return {qtype: QUESTIONS.get(qtype) or PARAPHRASES[qtype]}


def readout(qtype: str, answer: dict) -> tuple[float, float]:
    """Typed answer -> (rank_score, p_relevant)."""
    if qtype == "choice3":
        p = answer["probabilities"]
        p_rel, p_part = p.get("relevant", 0.0), p.get("partially", 0.0)
        return (2 * p_rel + p_part) / 2 + 1e-6 * p_rel, p_rel
    if qtype.startswith("noul"):  # noul and its paraphrases
        return answer["noul"], answer["noul"]
    if qtype == "score4":
        p = {int(k): v for k, v in answer["probabilities"].items()}
        return answer["score"] / 3, p.get(2, 0.0) + p.get(3, 0.0)
    raise ValueError(qtype)


def p_lenient(qtype: str, answer: dict) -> float:
    """P(relevant) when the qrels count partial relevance (grade >= 1) as relevant:
    choice3 -> P(relevant) + P(partially); score4 -> P(level >= 1); noul -> P(yes)."""
    if qtype == "choice3":
        p = answer["probabilities"]
        return p.get("relevant", 0.0) + p.get("partially", 0.0)
    if qtype == "score4":
        p = {int(k): v for k, v in answer["probabilities"].items()}
        return 1.0 - p.get(0, 0.0)
    if qtype.startswith("noul"):
        return answer["noul"]
    raise ValueError(qtype)
