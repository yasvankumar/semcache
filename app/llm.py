"""One interface for all models. LiteLLM for real calls; a deterministic mock for offline runs.

Mock behaviour (so routing/escalation can be demonstrated without API keys):
  * large model: answers using every retrieved passage's first sentence
  * small model: answers using the top passage only, and says "I don't know" on reasoning-style questions
"""
import re
import time
from dataclasses import dataclass

COMPLEX_WORDS = re.compile(r"\b(why|compare|comparison|difference|differ|versus|vs|explain|trade-?off)\b", re.I)


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


def _tokens(s: str) -> int:
    return max(1, int(len(s.split()) * 1.3))


def _first_sentence(s: str) -> str:
    return re.split(r"(?<=[.!?])\s", s.strip())[0]


def _mock(model: str, system: str, user: str, cfg) -> str:
    if "CLASSIFY" in system:
        return "COMPLEX" if COMPLEX_WORDS.search(user) else "SIMPLE"
    ctx = re.findall(r"^\[([\w-]+)\] (.+)$", user, flags=re.M)
    question = (re.search(r"^Question: (.+)$", user, flags=re.M) or [None, ""])[1]
    if not ctx:
        return "I don't know."
    if model == cfg.small_model:
        if COMPLEX_WORDS.search(question):
            return "I don't know based on the provided context."
        return _first_sentence(ctx[0][1]) + f" [{ctx[0][0]}]"
    return " ".join(_first_sentence(t) + f" [{i}]" for i, t in ctx)


def complete(model: str, system: str, user: str, cfg) -> LLMResponse:
    t0 = time.perf_counter()
    if cfg.llm_mode == "mock":
        time.sleep((0.15 if model == cfg.small_model else 0.6) * cfg.mock_latency_scale)
        text = _mock(model, system, user, cfg)
        in_tok, out_tok = _tokens(system + user), _tokens(text)
    else:
        import litellm
        r = litellm.completion(model=model, messages=[{"role": "system", "content": system},
                                                      {"role": "user", "content": user}])
        text = r.choices[0].message.content or ""
        in_tok, out_tok = r.usage.prompt_tokens, r.usage.completion_tokens
    return LLMResponse(text, in_tok, out_tok, (time.perf_counter() - t0) * 1000)
