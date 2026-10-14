# Benchmark (105 requests per run, threshold=0.95, embedder=sentence-transformers/all-MiniLM-L6-v2, llm=mock)

| Metric | Baseline (always large) | Optimized | Change |
|---|---|---|---|
| Cost / request | $0.00087 | $0.00008 | **91.0% lower** |
| Cache hit rate | - | 60.0% | |
| Escalation rate | - | 0.0% | |
| p50 latency | 44.2 ms | 26.5 ms | -40.0% |
| p95 latency | 60.2 ms | 242.6 ms | +302.7% |
| Quality (key-fact recall) | 1.000 | 1.000 | +0.000 |

Route split (optimized): {'simple': 34, 'complex': 8, 'cache': 63}
