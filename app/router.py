"""Rule-based router (+ optional tiny-LLM classifier) and the escalation check."""
import re
from . import llm

REASONING = re.compile(r"\b(compare|comparison|versus|vs|why|explain|difference|differ|trade-?off|"
                       r"pros and cons|how does .* work|should i)\b", re.I)


def complexity_score(query: str, retrieval_scores: list[float]) -> int:
    score = 0
    words = query.split()
    if len(words) > 14:
        score += 1
    if REASONING.search(query):
        score += 2
    entities = [w for w in words[1:] if w[0].isupper() or w.isdigit()]
    if len(entities) >= 2:
        score += 1
    # flat retrieval scores = no single passage clearly answers it => probably needs synthesis
    if len(retrieval_scores) >= 3 and (retrieval_scores[0] - retrieval_scores[2]) < 0.03:
        score += 1
    return score


def classify_rules(query, retrieval_scores, cfg) -> str:
    return "complex" if complexity_score(query, retrieval_scores) >= cfg.complex_cutoff else "simple"


def classify_llm(query, cfg) -> tuple[str, float, float]:
    """Returns (route, cost, latency_ms) - the classifier call itself costs money; count it."""
    r = llm.complete(cfg.small_model, "CLASSIFY the question as SIMPLE or COMPLEX. Reply with one word.",
                     query, cfg)
    route = "complex" if "COMPLEX" in r.text.upper() else "simple"
    return route, cfg.cost(cfg.small_model, r.input_tokens, r.output_tokens), r.latency_ms


def classify(query, retrieval_scores, cfg):
    """Returns (route, extra_cost, extra_latency_ms)."""
    if cfg.router_type == "llm":
        return classify_llm(query, cfg)
    return classify_rules(query, retrieval_scores, cfg), 0.0, 0.0


def needs_escalation(result: dict, cfg) -> bool:
    scores = result.get("retrieval_scores") or [0.0]
    ans = (result.get("answer") or "").lower()
    return (scores[0] < cfg.min_retrieval_score or not result.get("sources")
            or "don't know" in ans or not ans.strip())
