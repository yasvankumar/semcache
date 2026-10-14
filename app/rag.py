"""Tiny RAG pipeline: retrieve top-k passages, then generate with a chosen model."""
import json
import numpy as np
from . import llm
from .config import ROOT

SYSTEM = "Answer using only the numbered passages. Cite passage ids in brackets. Say I don't know if unsure."


class Rag:
    def __init__(self, cfg, embedder, corpus_path=None):
        self.cfg, self.embedder = cfg, embedder
        self.docs = json.load(open(corpus_path or ROOT / "data" / "corpus.json"))
        self.reindex()

    def reindex(self):
        """Call after the corpus changes (and bump the corpus version!)."""
        self.doc_vecs = np.vstack([self.embedder.embed(d["text"]) for d in self.docs])

    def retrieve(self, qvec: np.ndarray):
        sims = self.doc_vecs @ qvec
        idx = np.argsort(-sims)[: self.cfg.top_k]
        return [(self.docs[i], float(sims[i])) for i in idx]

    def generate(self, query: str, hits, model: str) -> dict:
        ctx = "\n".join(f"[{d['id']}] {d['text']}" for d, _ in hits)
        r = llm.complete(model, SYSTEM, f"{ctx}\n\nQuestion: {query}", self.cfg)
        idk = "don't know" in r.text.lower()
        cited = [d["id"] for d, _ in hits if f"[{d['id']}]" in r.text]
        return {
            "answer": r.text,
            "sources": [] if idk else cited,
            "retrieval_scores": [round(s, 4) for _, s in hits],
            "model": model, "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
            "cost": self.cfg.cost(model, r.input_tokens, r.output_tokens),
            "latency_ms": r.latency_ms, "error": None,
        }
