"""The core request flow: cache -> route -> (escalate) -> store -> log."""
import time
from . import router
from .cache import SemanticCache, is_cacheable
from .config import Config, load_config
from .db import DB
from .embedder import get_embedder, normalize
from .rag import Rag


def quality_score(answer: str, expected: list[str] | None):
    """Fraction of expected key facts present in the answer (swap for an LLM judge in production)."""
    if not expected:
        return None
    a = answer.lower()
    return sum(k.lower() in a for k in expected) / len(expected)


class Service:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.db = DB(self.cfg.db_path)
        self.embedder = get_embedder(self.cfg.embedder_kind, self.cfg.embedding_model)
        self.rag = Rag(self.cfg, self.embedder)
        self.cache = SemanticCache(self.db, self.cfg.threshold, self.cfg.ttl_seconds)
        if self.db.get_meta("corpus_version") is None:
            self.db.set_meta("corpus_version", self.cfg.initial_corpus_version)

    # -- corpus versioning ---------------------------------------------------
    @property
    def corpus_version(self) -> str:
        return self.db.get_meta("corpus_version")

    def bump_corpus_version(self) -> str:
        """Call whenever documents are re-ingested. Old cache entries stop matching immediately."""
        n = int(self.corpus_version.lstrip("v") or 0) + 1
        self.db.set_meta("corpus_version", f"v{n}")
        self.rag.reindex()
        self.cache.purge_stale(self.corpus_version)
        return self.corpus_version

    # -- logging ----------------------------------------------------------------
    def _log(self, mode, query, tenant, hit, route, model, escalated, in_tok, out_tok, cost, lat, quality, sim):
        self.db.execute(
            "INSERT INTO requests(ts,mode,query,tenant,cache_hit,route,model,escalated,input_tokens,"
            "output_tokens,cost,latency_ms,quality,similarity) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), mode, query, tenant, int(hit), route, model, int(escalated), in_tok, out_tok,
             cost, lat, quality, sim))

    # -- main flow -------------------------------------------------------------------
    def answer(self, query: str, tenant: str | None = None, personalized: bool = False,
               mode: str = "optimized", expected: list[str] | None = None) -> dict:
        t0 = time.perf_counter()
        tenant_key = tenant or "public"
        qvec = self.embedder.embed(normalize(query))
        elapsed = lambda: (time.perf_counter() - t0) * 1000

        if mode == "baseline":  # always-large, no cache: the comparison point
            hits = self.rag.retrieve(qvec)
            res = self.rag.generate(query, hits, self.cfg.large_model)
            lat = elapsed()
            self._log("baseline", query, tenant_key, False, "large", res["model"], False,
                      res["input_tokens"], res["output_tokens"], res["cost"], lat,
                      quality_score(res["answer"], expected), None)
            return {**res, "route": "large", "cache_hit": False, "latency_ms": lat}

        sim = 0.0
        if not personalized:  # personalised answers are neither read from nor written to the cache
            lk = self.cache.lookup(qvec, tenant_key, self.corpus_version)
            sim = lk.score
            if lk.payload is not None:
                lat = elapsed()
                self._log("optimized", query, tenant_key, True, "cache", "cache", False, 0, 0, 0.0, lat,
                          quality_score(lk.payload["answer"], expected), sim)
                return {**lk.payload, "route": "cache", "cache_hit": True, "cost": 0.0,
                        "latency_ms": lat, "similarity": sim, "matched_query": lk.matched_query}

        hits = self.rag.retrieve(qvec)
        scores = [s for _, s in hits]
        route, extra_cost, _ = router.classify(query, scores, self.cfg)
        model = self.cfg.small_model if route == "simple" else self.cfg.large_model
        res = self.rag.generate(query, hits, model)
        total_cost = res["cost"] + extra_cost
        in_tok, out_tok, escalated = res["input_tokens"], res["output_tokens"], False
        if route == "simple" and router.needs_escalation(res, self.cfg):
            escalated = True
            res2 = self.rag.generate(query, hits, self.cfg.large_model)
            total_cost += res2["cost"]
            in_tok += res2["input_tokens"]
            out_tok += res2["output_tokens"]
            res = res2
        res["cost"] = total_cost

        if is_cacheable(res, personalized):
            payload = {k: res[k] for k in ("answer", "sources", "retrieval_scores", "model")}
            self.cache.store(query, qvec, payload, tenant_key, self.corpus_version)

        lat = elapsed()
        self._log("optimized", query, tenant_key, False, route, res["model"], escalated, in_tok, out_tok,
                  total_cost, lat, quality_score(res["answer"], expected), sim)
        return {**res, "route": route, "cache_hit": False, "escalated": escalated, "latency_ms": lat,
                "similarity": sim}
