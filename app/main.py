"""FastAPI service.  Run: uvicorn app.main:app --reload"""
from fastapi import FastAPI
from pydantic import BaseModel
from app.service import Service
from huggingface_hub import login

login("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")


app = FastAPI(title="Semantic cache + cost-aware router")
svc = Service()


class Ask(BaseModel):
    query: str
    tenant: str | None = None
    personalized: bool = False
    mode: str = "optimized"   # optimized | baseline


@app.post("/ask")
def ask(req: Ask):
    return svc.answer(req.query, tenant=req.tenant, personalized=req.personalized, mode=req.mode)


@app.post("/ingest")
def ingest():
    """Pretend documents were re-ingested: bumps the corpus version so old cached answers die."""
    return {"corpus_version": svc.bump_corpus_version()}


@app.get("/health")
def health():
    return {"corpus_version": svc.corpus_version, "cache_entries": len(svc.cache)}
