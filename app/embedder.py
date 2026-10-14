"""Embeddings. sentence-transformers when available; a deterministic hashing embedder otherwise
(so the whole project runs offline). Vectors are always L2-normalised => dot product = cosine."""
import re
import zlib
import numpy as np


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


class HashEmbedder:
    """Bag of words + char trigrams hashed into a fixed vector. Crude, but deterministic and offline."""
    name = "hash-1024"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = normalize(text)
        for w in t.split():
            v[zlib.crc32(("w:" + w).encode()) % self.dim] += 2.0
        padded = f" {t} "
        for i in range(len(padded) - 2):
            v[zlib.crc32(("c:" + padded[i:i + 3]).encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v


class STEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True).astype(np.float32)


def get_embedder(kind: str = "auto", model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if kind in ("st", "auto"):
        try:
            return STEmbedder(model)
        except Exception as e:  # not installed / no network for model download
            if kind == "st":
                raise
            print(f"[embedder] falling back to hash embedder ({type(e).__name__})")
    return HashEmbedder()
