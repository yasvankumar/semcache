"""Router accuracy against hand-labelled queries. Usage: python -m scripts.eval_router [--router rules|llm]"""
import argparse, json
from app.config import ROOT, load_config
from app.embedder import get_embedder, normalize
from app.rag import Rag
from app import router


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--router", default=None); a = ap.parse_args()
    cfg = load_config()
    if a.router: cfg.router_type = a.router
    emb = get_embedder(cfg.embedder_kind, cfg.embedding_model); rag = Rag(cfg, emb)
    data = json.load(open(ROOT / "data/router_labels.json"))
    ok, cost = 0, 0.0
    for d in data:
        scores = [s for _, s in rag.retrieve(emb.embed(normalize(d["query"])))]
        route, c, _ = router.classify(d["query"], scores, cfg)
        ok += route == d["label"]; cost += c
        if route != d["label"]: print(f"MISS  want={d['label']:7} got={route:7} {d['query']}")
    print(f"router={cfg.router_type}  accuracy={ok}/{len(data)} = {ok/len(data):.1%}  classifier cost=${cost:.6f}")


if __name__ == "__main__":
    main()
