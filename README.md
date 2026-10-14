# Semantic caching + cost-aware routing

A RAG-style QA service that (1) returns cached answers for semantically similar queries, (2) sends easy
questions to a small model and hard ones to a large model, (3) escalates weak small-model answers, and
(4) proves the savings with a benchmark and a Streamlit dashboard.

```
query -> normalize -> embed -> cache lookup (tenant + corpus_version + TTL, sim >= threshold) -> HIT: return, $0
                                   | miss
                         retrieve top-k -> route (rules | tiny LLM) -> small or large model
                                   | small answer weak? (low retrieval score / no citation / "I don't know")
                                   +--> rerun on large model ("escalation")
                         store in cache (never errors / empty / "I don't know" / personalised) -> log request
```

## Quick start (works offline, no API keys)
```bash
pip install -r requirements.txt        # sentence-transformers + litellm are optional for the offline demo
python -m pytest -q                    # 9 tests: cache, invalidation, tenancy, TTL, escalation, router
python -m scripts.tune_threshold       # step 4: sweep 0.85-0.97 on paraphrase vs near-miss pairs (--write saves it)
python -m scripts.eval_router          # step 5: router accuracy on labelled queries (--router llm to compare)
python -m scripts.benchmark            # step 8: baseline vs optimized -> results/benchmark.md
streamlit run dashboard/app.py         # step 7: dashboard (screenshot it for your README)
uvicorn app.main:app --reload          # API: POST /ask, POST /ingest (bumps corpus_version), GET /health
```
The dashboard and benchmark read the same SQLite log (`data/app.db`).

## Going from mock to real
1. `config.yaml -> llm.mode: litellm`, set provider keys, set `models.small/large` and the price table.
2. Install `sentence-transformers` (embedder `auto` picks it up). **Re-run `tune_threshold --write`** - the
   threshold depends on the embedding model. The offline hash embedder is lexical, so it gets ~0% paraphrase
   hits; real embeddings are what make paraphrase caching work.
3. Replace `data/corpus.json` with your documents; call `Service.bump_corpus_version()` after every re-ingest.
4. Extend the seed data to the sizes the brief asks for: `data/router_labels.json` (25 -> 100 labelled queries),
   `data/eval_set.json` (add questions from your domain), and the pair files (templates -> real traffic).
5. Replace `quality_score` (key-fact recall) with an LLM judge or human-labelled answers.
6. Optional: swap `app/cache.py`'s numpy index for Qdrant/Redis - keep the tenant/version/TTL filters in the query.

## Layout
| Path | What |
|---|---|
| `app/service.py` | The request flow from the brief, plus baseline mode and logging |
| `app/cache.py` | Semantic cache, validity rules (tenant, corpus_version, TTL), `is_cacheable` |
| `app/router.py` | Rule-based complexity score, LLM classifier, escalation check |
| `app/llm.py` | LiteLLM wrapper + deterministic mock so everything runs offline |
| `app/rag.py` | Retrieval + generation |
| `app/db.py` | SQLite request log / cache entries / corpus version |
| `scripts/` | threshold tuning, router eval, benchmark |
| `dashboard/app.py` | Streamlit dashboard |

## Sample result (offline mock LLM + hash embedder - illustrative only)
See `results/benchmark.md`. These numbers come from a mock model and a toy corpus, so **do not put them on a
resume**. Re-run with real models and report those. Things worth noting in your write-up:
* Cost drops mostly from cache hits on repeat traffic; benchmarking on unique queries would hide that.
* p95 can get *worse*: an escalated query pays for the small call and then the large call. Report it honestly.
* The first benchmark run exposed a false cache hit ("refund policy in the US" served the EU answer at 0.91
  similarity) because the near-miss set lacked domain siblings. Near-miss pairs from *your* domain matter most.

## Design notes
* **Invalidation:** `corpus_version` on every entry + TTL; `bump_corpus_version()` also purges old entries.
* **Safety:** personalised requests skip the cache entirely; entries are keyed by tenant.
* **Cost accounting:** cost = tokens x price table; escalated requests pay for both calls; an LLM router's own
  call is counted. Embedding cost is ignored (negligible for local models).
