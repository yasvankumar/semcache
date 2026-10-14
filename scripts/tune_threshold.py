"""Sweep the similarity threshold on paraphrase pairs (should hit) and near-miss pairs (must NOT hit).
Picks the lowest threshold with ZERO false hits. Usage: python -m scripts.tune_threshold [--write]"""
import argparse, json, re
import numpy as np
from app.config import ROOT, load_config
from app.embedder import get_embedder, normalize


def sims(emb, pairs):
    return np.array([float(emb.embed(normalize(a)) @ emb.embed(normalize(b))) for a, b in pairs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write the chosen threshold into config.yaml")
    args = ap.parse_args()
    cfg = load_config()
    emb = get_embedder(cfg.embedder_kind, cfg.embedding_model)
    para = sims(emb, json.load(open(ROOT / "data/paraphrase_pairs.json")))
    near = sims(emb, json.load(open(ROOT / "data/near_miss_pairs.json")))
    print(f"embedder={getattr(emb, 'name', '?')}  paraphrase sim mean={para.mean():.3f}  near-miss sim max={near.max():.3f}\n")
    print("threshold | paraphrase hit rate | false hits (of %d near-misses)" % len(near))
    chosen = None
    for t in np.arange(0.85, 0.9701, 0.01):
        hit, fh = (para >= t).mean(), int((near >= t).sum())
        print(f"  {t:.2f}    |       {hit:6.1%}        |   {fh}")
        if chosen is None and fh == 0:
            chosen = round(float(t), 2)
    if chosen is None:
        print("\nNo threshold in 0.85-0.97 gives zero false hits. Use a better embedder, add a cross-encoder "
              "re-check, or key entries on extracted entities (region, plan...).")
        return
    print(f"\nChosen threshold: {chosen}")
    if args.write:
        p = ROOT / "config.yaml"
        p.write_text(re.sub(r"(threshold:\s*)[\d.]+", rf"\g<1>{chosen}", p.read_text(), count=1))
        print("config.yaml updated")


if __name__ == "__main__":
    main()
