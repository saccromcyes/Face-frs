"""Match a face embedding against the enrolled gallery.

FaceNet (InceptionResnetV1) embeddings are compared by cosine distance (1 - cosine similarity):
0 means identical direction, 1 means unrelated. A face is a match only when its distance to the
closest enrolled identity is at most `threshold`; otherwise it is reported as unknown.

The previous version computed similarities but never applied the threshold (its check was a
`pass`), so every face matched somebody.
"""

from __future__ import annotations

import numpy as np


def cosine_distances(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Cosine distance from one embedding to each row of `gallery`."""
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    gallery = np.asarray(gallery, dtype=np.float32).reshape(len(gallery), -1)
    if gallery.size == 0:
        return np.zeros(0, dtype=np.float32)
    q = query / max(float(np.linalg.norm(query)), 1e-12)
    g = gallery / np.clip(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12, None)
    return 1.0 - g @ q


def match(query: np.ndarray, names: list[str], gallery: np.ndarray, threshold: float, top_k: int = 5) -> dict:
    """Return the best match (or None) plus the top-k candidates with their distances."""
    if len(names) == 0:
        return {"match": None, "distance": None, "similarity": None, "candidates": []}
    distances = cosine_distances(query, gallery)
    order = np.argsort(distances)[:top_k]
    candidates = [
        {"name": names[i], "distance": round(float(distances[i]), 4), "is_match": bool(distances[i] <= threshold)}
        for i in order
    ]
    best = candidates[0]
    return {
        "match": best["name"] if best["is_match"] else None,
        "distance": best["distance"],
        "similarity": round(1.0 - best["distance"], 4),
        "threshold": threshold,
        "candidates": candidates,
    }
