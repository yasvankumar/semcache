"""Config loading. Environment variables override config.yaml (handy for tests/CI)."""
import os
from dataclasses import dataclass
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    small_model: str
    large_model: str
    prices: dict
    threshold: float
    ttl_seconds: int
    initial_corpus_version: str
    router_type: str
    complex_cutoff: int
    min_retrieval_score: float
    top_k: int
    llm_mode: str
    mock_latency_scale: float
    embedder_kind: str
    embedding_model: str
    db_path: str

    def cost(self, model: str, in_tok: int, out_tok: int) -> float:
        p = self.prices[model]
        return in_tok / 1000 * p["input"] + out_tok / 1000 * p["output"]


def load_config(path: str | None = None, **overrides) -> Config:
    path = path or os.environ.get("CONFIG", str(ROOT / "config.yaml"))
    raw = yaml.safe_load(open(path))
    cfg = Config(
        small_model=raw["models"]["small"],
        large_model=raw["models"]["large"],
        prices=raw["prices_per_1k_tokens"],
        threshold=float(os.environ.get("CACHE_THRESHOLD", raw["cache"]["threshold"])),
        ttl_seconds=int(raw["cache"]["ttl_seconds"]),
        initial_corpus_version=raw["cache"]["initial_corpus_version"],
        router_type=raw["router"]["type"],
        complex_cutoff=int(raw["router"]["complex_cutoff"]),
        min_retrieval_score=float(raw["escalation"]["min_retrieval_score"]),
        top_k=int(raw["retrieval"]["top_k"]),
        llm_mode=os.environ.get("LLM_MODE", raw["llm"]["mode"]),
        mock_latency_scale=float(raw["llm"]["mock_latency_scale"]),
        embedder_kind=os.environ.get("EMBEDDER", raw["embedder"]["kind"]),
        embedding_model=raw["embedder"]["model"],
        db_path=os.environ.get("DB_PATH", str(ROOT / raw["db_path"])),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
