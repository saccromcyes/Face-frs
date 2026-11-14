import numpy as np

from app.matcher import cosine_distances, match


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


GALLERY = np.stack([unit([1, 0, 0]), unit([0, 1, 0]), unit([1, 1, 0])])
NAMES = ["asha", "ravi", "meera"]


def test_cosine_distance_basics():
    d = cosine_distances(unit([1, 0, 0]), GALLERY)
    assert d[0] == 0
    assert abs(d[1] - 1) < 1e-6
    assert abs(d[2] - (1 - 1 / np.sqrt(2))) < 1e-6


def test_close_face_matches():
    out = match(unit([0.98, 0.05, 0.0]), NAMES, GALLERY, threshold=0.45)
    assert out["match"] == "asha"
    assert out["candidates"][0]["is_match"]


def test_stranger_is_unknown_not_the_nearest_person():
    # The old matcher returned the nearest identity even when it was far away.
    out = match(unit([0, 0, 1]), NAMES, GALLERY, threshold=0.45)
    assert out["match"] is None
    assert out["distance"] == 1.0
    assert not any(c["is_match"] for c in out["candidates"])


def test_threshold_controls_the_decision():
    probe = unit([1, 0.6, 0])  # cosine distance ~0.14 to asha, ~0.03 to meera
    assert match(probe, NAMES, GALLERY, threshold=0.01)["match"] is None
    assert match(probe, NAMES, GALLERY, threshold=0.10)["match"] == "meera"


def test_empty_gallery():
    assert match(unit([1, 0, 0]), [], np.zeros((0, 3)), threshold=0.45)["match"] is None
